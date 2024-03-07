import argparse
import torch

from diffusers import StableVideoDiffusionPipeline

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="no-traffic-encoded")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    return parser


def get_models():
    enc_dec_dtype, variant = torch.float16, "fp16"
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=enc_dec_dtype, variant=variant 
    )
    pipe.enable_model_cpu_offload()
    image_processor = pipe.image_processor
    vae = pipe.vae
    del pipe
    for p in vae.parameters():
        p.requires_grad = False
    print('Loaded encoder and decoder.')
    return image_processor, vae


def load_video(fname):
    return torch.load(fname)


def encode(video, image_processor, vae, chunk_size):
    video = image_processor.preprocess((video/255).permute(0, 3, 1, 2))
    def encode_chunk(chunk):
        chunk = torch.nn.functional.interpolate(chunk, scale_factor=2)
        dist = vae.encode(chunk.to(vae.dtype).cuda()).latent_dist
        return dist.mean, dist.std
    outputs = [encode_chunk(video[i:i+chunk_size]) for i in range(0, video.shape[0], chunk_size)]
    means, stdevs = torch.cat([o[0] for o in outputs]), torch.cat([o[1] for o in outputs])
    return means.cpu(), stdevs.cpu()


def save(fname, tensor):
    torch.save(tensor, fname)


def main(args):
    args = create_argparser().parse_args()
    path = args.path
    image_processor, vae = get_models()

    mean_x, mean_x2, n_obs = torch.zeros(1, 4, 1, 1), torch.zeros(1, 4, 1, 1), 0
    for mode in ['train', 'test']:
        split_path = path + f"/video_{mode}.csv"
        fnames = [line.rstrip('\n').split('/')[-1] for line in open(split_path, 'r').readlines() if '.pt' in line]
        # fnames = fnames[:3]

        for fname in fnames:
            video = load_video(path + "/" + fname)
            encoded_means, encoded_stdevs = encode(video, image_processor, vae, args.chunk_size)
            save(path + "/encoded_" + fname, encoded_means)
            
            if args.normalize and mode == 'train':  # accumulate training data channel-wise statistics
                n_obs_curr = encoded_means[:, 0].numel()
                mean_x_curr = encoded_means.to(torch.float64).mean(dim=(0, 2, 3), keepdim=True)
                mean_x = n_obs/(n_obs+n_obs_curr) * mean_x + n_obs_curr/(n_obs+n_obs_curr) * mean_x_curr
                mean_x2_curr = (encoded_means**2).to(torch.float64).mean(dim=(0, 2, 3), keepdim=True)
                mean_x2 = n_obs/(n_obs+n_obs_curr) * mean_x2 + n_obs_curr/(n_obs+n_obs_curr) * mean_x2_curr
                n_obs = n_obs + n_obs_curr
                print(f"n_obs: {n_obs}, mean_x: {mean_x.flatten()}, mean_x2: {mean_x2.flatten()}")

        if args.normalize:
            if mode == 'train':  # compute training data channel-wise mean and std
                mean = mean_x.to(encoded_means.dtype)
                std = torch.sqrt(mean_x2 - mean_x**2).to(encoded_means.dtype)
                stats_dict = {"mean": mean.flatten(), "std": std.flatten(), "n_obs": n_obs}
                print("=== Final Stats ===")
                print(stats_dict)
                torch.save(stats_dict, path + f"/encoded_{mode}_norm_stats.pt")

            print(f"Normalizing {mode} data.")
            for fname in fnames:
                encoded_path = path + "/encoded_" + fname
                video = load_video(encoded_path)
                normalized_video = (video - mean) / (std+1e-8)
                save(encoded_path, normalized_video.to(video.dtype))


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)

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
        chunk = torch.nn.functional.interpolate(chunk, scale_factor=2).to(vae.dtype).cuda()
        dist = vae.encode(chunk).latent_dist
        print(f"Means: {dist.mean.mean():.4f} / {dist.std.mean():.4f} --- Stdev: {dist.mean.std():.4f} / {dist.std.std():.4f}")
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

    for mode in ['train', 'test']:
        split_path = path + f"/video_{mode}.csv"
        fnames = [line.rstrip('\n').split('/')[-1] for line in open(split_path, 'r').readlines() if '.pt' in line]

        sum_x, sum_x2, n_obs = torch.zeros(1, 4, 1, 1), torch.zeros(1, 4, 1, 1), 0

        for fname in fnames:
            video = load_video(path + "/" + fname)
            encoded_means, encoded_stdevs = encode(video, image_processor, vae, args.chunk_size)
            save(path + "/encoded_" + fname, encoded_means)
            
            # record channel-wise normalization statistics
            if args.normalize:
                sum_x += encoded_means.to(torch.float64).sum(dim=(0, 2, 3), keepdim=True)
                sum_x2 += (encoded_means**2).to(torch.float64).sum(dim=(0, 2, 3), keepdim=True)
                n_obs += encoded_means[0].numel()
                print(sum_x2)

        mean = (sum_x/n_obs).to(encoded_means.dtype)
        std = torch.sqrt(sum_x2/n_obs - mean**2).to(encoded_means.dtype)

        if args.normalize:
            print(f"Normalizing {mode} data.")
            for fname in fnames:
                encoded_path = path + "/encoded_" + fname
                video = load_video(encoded_path)
                normalized_video = (video - mean) / (std+1e-8)
                save(encoded_path, normalized_video)


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)

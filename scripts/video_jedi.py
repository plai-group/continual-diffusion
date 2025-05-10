import torch as th
import numpy as np
import argparse
import os
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm

from videojedi import JEDiMetric

# Metrics
from improved_diffusion.video_datasets import get_eval_dataset, eval_dataset_configs
from improved_diffusion.test_util import parse_eval_run_identifier, Protect


class SampleDataset(th.utils.data.Dataset):
    def __init__(self, samples_path, sample_idx, length, start_idx=0):
        self.samples_path = Path(samples_path)
        self.start_idx = start_idx
        self.sample_idx = sample_idx
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = self.samples_path / f"sample_{self.start_idx+idx:04d}-{self.sample_idx}.npy"
        npy = np.load(path).astype(np.float32)
        normed = -1 + 2 * npy / 255
        return th.tensor(normed).type(th.float32), {}


class DecodedDataset(th.utils.data.Dataset):
    def __init__(self, encoded_dataset, cache_path, decode_chunk_size,
                 pre_decode=False, subset_indices=None):
        self.encoded_dataset = encoded_dataset
        self.cache_path = Path(cache_path)
        self.decode_chunk_size = decode_chunk_size
        self.vae = None
        if pre_decode:
            self.pre_decode(subset_indices)

    def __len__(self):
        return len(self.encoded_dataset)

    def __getitem__(self, idx):
        path = self.cache_path / f"sample_{idx:04d}.npy"
        with Protect(path, timeout=3600):
            if not path.exists():
                print(f"Decoding data item {idx}...")
                encoding, _ = self.encoded_dataset[idx]
                video = self._decode(encoding)
                np.save(path, video)
                print(f"Finished decoding data item {idx}.")
        npy = np.load(path).astype(np.float32)
        normed = -1 + 2 * npy / 255
        return th.tensor(normed).type(th.float32), {}

    @th.no_grad()
    def _decode(self, encoding):
        no_vae = self.vae is None
        if no_vae:
            self._initialize_vae()
        with th.no_grad():
            decoded = [self.vae.decode(encoding[j:j+self.decode_chunk_size].to(self.vae.device)/0.13025
                       ).sample for j in range(0, encoding.shape[0], self.decode_chunk_size)]
        drange = [-1, 1]
        decoded = th.cat(decoded, dim=0).cpu().clamp(*drange).numpy()
        decoded = (decoded - drange[0]) / (drange[1] - drange[0]) * 255
        if no_vae:
            self._remove_vae()
        return decoded.astype(np.uint8)

    def pre_decode(self, subset_indices=None):
        self._initialize_vae()
        init_indices = [i for i in range(len(self))] if subset_indices is None else subset_indices
        for i in np.random.permutation(init_indices):  # random ordering
            self[i]
        self._remove_vae()

    def _initialize_vae(self):
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=th.float16)
        self.vae.eval()
        if th.cuda.is_available():
            self.vae = self.vae.cuda()

    def _remove_vae(self):
        del self.vae
        self.vae = None
        import gc
        gc.collect()
        th.cuda.empty_cache()


def compute_jedi(jedi_feature_path, test_dataset, sample_dataset, num_videos, batch_size=16):
    def transform_collate(batch):
        return th.stack([(item[0]+1)/2 for item in batch], dim=0), {}

    truth_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=transform_collate)
    sample_loader = th.utils.data.DataLoader(sample_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=transform_collate)

    jedi = JEDiMetric(feature_path=jedi_feature_path, model_dir='.')
    jedi.load_features(truth_loader, sample_loader, num_samples=num_videos)
    jedi_val = jedi.compute_metric()
    return jedi_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--num_videos", type=int, default=None,
                        help="Number of generated samples per test video.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for extracting video features the I3D model.")
    parser.add_argument("--sample_idx", type=int, default=0, help="sample seed")
    parser.add_argument("--decode_chunk_size", type=int, default=5)
    parser.add_argument("--decode_cache_dir", type=str, default="./tmp/decoded_ground_truth")
    args = parser.parse_args()

    parsed = parse_eval_run_identifier(os.path.basename(args.eval_dir))
    T, obs_length, eval_on_train = parsed["T"], parsed["n_obs"], parsed["eval_on_train"]
    lower_frame_range, upper_frame_range = parsed["lower_frame_range"], parsed["upper_frame_range"]
    eval_dataset_config = parsed["eval_dataset_config"]
    if eval_dataset_config == eval_dataset_configs["default"]:
        assert args.num_videos is not None, "You must specified how many videos were generated by video_sample.py if not using visualization mode."

    jedi_save_path = Path(args.eval_dir) / f"jedi-{args.num_videos}-{args.sample_idx}.txt"
    if jedi_save_path.exists():
        jedi = np.loadtxt(jedi_save_path).squeeze()
        print(f"JEDi is already computed: {jedi}")
        exit()

    # Load model args
    model_args_path = Path(args.eval_dir) / "model_config.json"
    with open(model_args_path, "r") as f:
        model_args = argparse.Namespace(**json.load(f))

    # Load the dataset (to get observations from)
    eval_dataset_args = dict(dataset_name=model_args.dataset, T=T, train=eval_on_train, spacing_kwargs=dict(n_data=args.num_videos),
                             eval_dataset_config=eval_dataset_config, frame_range=(lower_frame_range, upper_frame_range))
    test_dataset_full = get_eval_dataset(**eval_dataset_args)
    sample_dataset = SampleDataset(samples_path=(Path(args.eval_dir) / "samples"), sample_idx=args.sample_idx, length=args.num_videos)

    # Set batch size given dataset if not specified
    if args.batch_size is None:
        args.batch_size = {'mazes_cwvae': 16, 'minerl': 8, 'carla_no_traffic': 4, 'carla_no_traffic_2x': 4, 'carla_no_traffic_2x_encoded': 4}[args.dataset]

    # Decode ground truth data and save them if it is encoded
    encoded_test_data = test_dataset_full[0][0].shape != sample_dataset[0][0].shape
    subset_indices = list(range(args.num_videos))
    if encoded_test_data:
        dataset_name = model_args.dataset[10:] if model_args.dataset.startswith("streaming_") else model_args.dataset
        cache_dir = Path(args.decode_cache_dir) / f"{dataset_name}_{T}_{eval_dataset_config}_{lower_frame_range}_{upper_frame_range}_{eval_on_train}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        test_dataset_full = DecodedDataset(test_dataset_full, cache_dir, args.decode_chunk_size,
                                           pre_decode=True, subset_indices=subset_indices) 
    test_dataset = th.utils.data.Subset(
        dataset=test_dataset_full,
        indices=subset_indices,
    )

    jedi_feature_path = Path(args.eval_dir) / f"jedi_features_{args.sample_idx}"
    jedi_feature_path.mkdir(parents=True, exist_ok=True)

    jedi_val = compute_jedi(jedi_feature_path, test_dataset, sample_dataset, num_videos=args.num_videos, batch_size=args.batch_size)
    np.savetxt(jedi_save_path, np.array([jedi_val]))
    print(f"JEDI: {jedi_val}")

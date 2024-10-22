import torch as th
import numpy as np
import argparse
import os
from pathlib import Path
import json
from collections import defaultdict
import tensorflow.compat.v1 as tf

# Metrics
from improved_diffusion.video_datasets import get_test_dataset, get_vis_dataset
import improved_diffusion.frechet_video_distance as fvd
from improved_diffusion import test_util
from improved_diffusion.script_util import str2bool

tf.disable_eager_execution() # Required for our FVD computation code


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
        if not path.exists():
            print(f"Decoding data item {idx}...")
            encoding, _ = self.encoded_dataset[idx]
            video = self._decode(encoding)
            np.save(path, video)
            print(f"Finished decoding data item {idx}.")
        npy = np.load(path).astype(np.float32)
        # return npy, {}
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
        # return decoded
        return decoded.astype(np.uint8)

    def pre_decode(self, subset_indices=None):
        self._initialize_vae()
        init_indices = range(len(self)) if subset_indices is None else subset_indices
        for i in init_indices:
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


class FVD:
    def __init__(self, batch_size, T, frame_shape):
        self.batch_size = batch_size
        self.vid = tf.placeholder("uint8", [self.batch_size, T, *frame_shape])
        self.vid_feature_vec = fvd.create_id3_embedding(fvd.preprocess(self.vid, (224, 224)), batch_size=self.batch_size)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())

    def extract_features(self, vid):
        def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
            # From here: https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
            pad_size = target_length - array.shape[axis]
            if pad_size <= 0:
                return array
            npad = [(0, 0)] * array.ndim
            npad[axis] = (0, pad_size)
            return np.pad(array, pad_width=npad, mode='constant', constant_values=0)
        # vid is expected to have a shape of BxTxCxHxW
        B = vid.shape[0]
        vid = np.moveaxis(vid, 2, 4)  # B, T, H, W, C
        vid = pad_along_axis(vid, target_length=self.batch_size, axis=0)
        features = self.sess.run(self.vid_feature_vec, feed_dict={self.vid: vid})
        features = features[:B]
        return features

    @staticmethod
    def compute_fvd(vid1_features, vid2_features):
        return fvd.fid_features_to_metric(vid1_features, vid2_features)


def compute_fvd(test_dataset, sample_dataset, T, num_videos, batch_size=16):
    _, C, H, W = sample_dataset[0][0].shape
    fvd_handler = FVD(batch_size=batch_size, T=T, frame_shape=[H, W, C])
    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    sample_loader = th.utils.data.DataLoader(sample_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    assert len(test_dataset) == num_videos, f"{len(test_dataset)} != {num_videos}"
    assert len(sample_dataset) == num_videos, f"{len(sample_dataset)} != {num_videos}"
    with tf.Graph().as_default():
        all_test_features = []
        all_pred_features = []
        for (test_batch, _), (sample_batch, _) in zip(test_loader, sample_loader):
            scale = lambda x: ((x.numpy()+1)*255/2).astype(np.uint8)  # scale from [-1, 1] to [0, 255]
            test_batch = scale(test_batch)
            sample_batch = scale(sample_batch)
            test_features = fvd_handler.extract_features(test_batch)
            sample_features = fvd_handler.extract_features(sample_batch)
            all_test_features.append(test_features)
            all_pred_features.append(sample_features)
        all_test_features = np.concatenate(all_test_features, axis=0)
        all_pred_features = np.concatenate(all_pred_features, axis=0)
        fvd = fvd_handler.compute_fvd(all_pred_features, all_test_features)
    return fvd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--num_videos", type=int, default=None,
                        help="Number of generated samples per test video.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for extracting video features the I3D model.")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--T", type=int, default=None, help="Length of the videos. If not specified, it will be inferred from the dataset.")
    parser.add_argument("--eval_on_train", type=str2bool, default=False)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--visualize_mode", type=str2bool, default=False, help="Used to produce videos of specific video subsequences.")
    parser.add_argument("--decode_chunk_size", type=int, default=5)
    parser.add_argument("--decode_cache_dir", type=str, default="./tmp/plaicraft_decoded")
    args = parser.parse_args()

    if args.eval_on_train:
        save_path = Path(args.eval_dir) / f"fvd-{args.num_videos}-{args.sample_idx}-train.txt"
    else:
        save_path = Path(args.eval_dir) / f"fvd-{args.num_videos}-{args.sample_idx}.txt"

    if save_path.exists():
        fvd = np.loadtxt(save_path).squeeze()
        print(f"FVD already computed: {fvd}")
        exit()

    # Load model args
    model_args_path = Path(args.eval_dir) / "model_config.json"
    with open(model_args_path, "r") as f:
        model_args = argparse.Namespace(**json.load(f))

    # Set batch size given dataset if not specified
    args.dataset = model_args.dataset
    if args.batch_size is None:
        args.batch_size = {'mazes_cwvae': 16, 'minerl': 8, 'carla_no_traffic': 4, 'carla_no_traffic_2x': 4, 'carla_no_traffic_2x_encoded': 4}[args.dataset]

    if args.T is None:
        args.T = model_args.T

    samples_prefix = "samples_train" if args.eval_on_train else "samples"
    if args.visualize_mode:
        samples_prefix += "_vis"
        sample_dataset = SampleDataset(samples_path=(Path(args.eval_dir) / samples_prefix), sample_idx=args.sample_idx,
                                       length=args.num_videos, start_idx=args.start_idx)
        test_dataset_full = get_vis_dataset(dataset_name=args.dataset, T=args.T)
    else:
        sample_dataset = SampleDataset(samples_path=(Path(args.eval_dir) / samples_prefix), sample_idx=args.sample_idx, length=args.num_videos)
        test_dataset_full = get_test_dataset(dataset_name=args.dataset, T=args.T, n_data=args.num_videos)

    if args.eval_on_train:
        test_dataset_full.set_train()
    encoded_test_data = test_dataset_full[0][0].shape != sample_dataset[0][0].shape
    subset_indices = list(range(args.start_index, args.start_index+args.num_videos))
    if encoded_test_data:
        cache_dir = Path(args.decode_cache_dir) / str(test_dataset_full.__class__.__name__)
        cache_dir = cache_dir / "train" if args.eval_on_train else cache_dir / "test"
        cache_dir.mkdir(parents=True, exist_ok=True)
        test_dataset_full = DecodedDataset(test_dataset_full, cache_dir, args.decode_chunk_size,
                                           pre_decode=True, subset_indices=subset_indices)
    test_dataset = th.utils.data.Subset(
        dataset=test_dataset_full,
        indices=subset_indices,
    )

    fvd = compute_fvd(test_dataset, sample_dataset, T=args.T, num_videos=args.num_videos, batch_size=args.batch_size)
    np.savetxt(save_path, np.array([fvd]))
    print(f"FVD: {fvd}")

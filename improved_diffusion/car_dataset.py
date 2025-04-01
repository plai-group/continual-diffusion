import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import ffmpeg
import os
import time
import tqdm


class ContinuousCarDataset(Dataset):
    LATENT_FPS = 20
    LATENTS_PER_BATCH = 100
    USE_FP16 = True

    def __init__(self, dataset_path, window_length=100, output_fps=20, frame_range=(0, None)):
        self.train_path = os.path.join(dataset_path, 'train')
        self.test_path = os.path.join(dataset_path, 'test')
        self.is_test = False

        self.window_length = self.T = window_length
        self.output_fps = output_fps
        self.original_frame_range = frame_range

        self._validate_parameters()
        self._initialize_file_index_mapping()

    def _validate_parameters(self):
        assert isinstance(self.window_length, int) and self.window_length > 0, f"window_length must be a positive integer, but got {self.window_length}."
        assert isinstance(self.output_fps, int) and self.output_fps > 0, f"output_fps must be a positive integer, but got {self.output_fps}."
        assert 1 <= self.output_fps <= self.LATENT_FPS, f"output_fps must be between 1 and {self.LATENT_FPS}, but got {self.output_fps}."

    def _initialize_file_index_mapping(self):
        pt_files = self.dataset_path.rglob('encoded_video/batch_*.pt')
        def sort_fn(k):
            split = str(k).split('/')
            video_name, filename = split[-3], split[-1]
            video_name_split = video_name.split('--')
            primary_id, secondary_id, batch_id = video_name_split[0], int(video_name_split[2]), int(filename[6:10])
            return primary_id, secondary_id, batch_id
        pt_files = sorted(pt_files, key=sort_fn)
        if not pt_files:
            raise ValueError(f"No .pt files found in {self.dataset_path}")

        self.pt_files = pt_files
        self.pt_file_boundaries = []
        self.total_frames = 0

        for pt_file in tqdm.tqdm(self.pt_files, desc="Retrieving file info..."):
            batch_data = torch.load(pt_file, map_location='cpu')
            num_frames_in_batch = batch_data['quantized_latents'].shape[0]
            start_frame = self.total_frames
            end_frame = self.total_frames + num_frames_in_batch
            self.pt_file_boundaries.append((start_frame, end_frame))
            self.total_frames = end_frame

        self.frame_range = self.original_frame_range
        if self.original_frame_range[1] is None or self.original_frame_range[1] > self.total_frames:
            self.frame_range = (self.original_frame_range[0], self.total_frames)
        print(f"Original Frame Range ({self.original_frame_range}) => Updated Frame Range ({self.frame_range})")

    def _dequantize_from_int8(self, quantized_tensor, min_val, scale):
        while min_val.dim() < quantized_tensor.dim():
            min_val = min_val.unsqueeze(-1)
        while scale.dim() < quantized_tensor.dim():
            scale = scale.unsqueeze(-1)
        dequantized = quantized_tensor.to(torch.float32) * scale + min_val
        return dequantized.half() if self.USE_FP16 else dequantized

    def __len__(self):
        return (self.frame_range[1] - self.frame_range[0]) - (self.window_length - 1)

    def _get_start_frame_index(self, idx):
        return self.frame_range[0] + idx

    def __getitem__(self, idx):
        start_frame = self._get_start_frame_index(idx)
        end_frame = start_frame + self.window_length

        if end_frame > self.total_frames:
            raise IndexError(f"Index {idx} out of range.")

        frames_needed = self.window_length
        current_frame = start_frame
        frames = []

        while frames_needed > 0:
            for pt_file_idx, (pt_start_frame, pt_end_frame) in enumerate(self.pt_file_boundaries):
                if pt_start_frame <= current_frame < pt_end_frame:
                    break
            else:
                raise ValueError(f"Frame {current_frame} not found in any pt file.")

            pt_file = self.pt_files[pt_file_idx]
            batch_data = torch.load(pt_file)
            quantized_latents = batch_data['quantized_latents']
            min_vals = batch_data['min_vals']
            scales = batch_data['scales']

            local_frame_idx = current_frame - pt_start_frame
            frames_available = quantized_latents.shape[0] - local_frame_idx
            frames_to_take = min(frames_available, frames_needed)

            sliced_latents = quantized_latents[local_frame_idx:local_frame_idx+frames_to_take]
            sliced_min_vals = min_vals[local_frame_idx:local_frame_idx+frames_to_take]
            sliced_scales = scales[local_frame_idx:local_frame_idx+frames_to_take]

            dequantized_latents = self._dequantize_from_int8(sliced_latents, sliced_min_vals, sliced_scales)
            frames.append(dequantized_latents)

            frames_needed -= frames_to_take
            current_frame += frames_to_take

        frames = torch.cat(frames, dim=0)
        absolute_index_map = torch.arange(start_frame, end_frame)

        if frames.shape[0] < self.window_length:
            raise IndexError(f"Incomplete window at index {idx}, should have been discarded.")

        return frames, absolute_index_map
        # return {
        #     "frames": frames,
        #     "absolute_index_map": absolute_index_map,
        # }

    @staticmethod
    def collate_fn(batch):
        return {
            "frames": torch.stack([item["frames"] for item in batch]),
            "absolute_index_map": torch.stack([item["absolute_index_map"] for item in batch]),
        }

    @property
    def dataset_path(self):
        return Path(self.test_path if self.is_test else self.train_path)

    def set_train(self):
        self.is_test = False
        self._initialize_file_index_mapping()
        print('setting train mode')

    def set_test(self):
        self.is_test = True
        self._initialize_file_index_mapping()
        print('setting test mode')


class ChunkedCarDataset(ContinuousCarDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_start_frame_index(self, idx):
        return self.frame_range[0] + idx * self.window_length

    def __len__(self):
        """Return the total number of windows across all sessions."""
        return (self.frame_range[1]-self.frame_range[0]) // self.window_length


class SpacedCarDataset(ContinuousCarDataset):
    def __init__(self, n_data: int, *args, **kwargs):
        self.n_data = n_data
        super().__init__(*args, **kwargs)

    def _get_start_frame_index(self, idx):
        return self.frame_range[0] + idx * self.spacing

    def __len__(self):
        """Return the total number of windows across all sessions."""
        return self.n_data

    def _initialize_file_index_mapping(self):
        super()._initialize_file_index_mapping()
        self.spacing = (self.frame_range[1]-self.frame_range[0]) // self.n_data
        assert self.spacing >= self.window_length,\
               f"Spacing to window length ratio is too small: {self.spacing} vs {self.window_length}"
        self.spacing = self.spacing - self.spacing % self.window_length  # spacing is divisible by window_length
        print(f"Total Frames: {self.total_frames}, Selected Frames: {self.frame_range[1]-self.frame_range[0]}, "+\
              f"Spacing: {self.spacing}, # Data: {self.n_data}")


if __name__ == "__main__":
    # dataset_path = "/ubc/cs/research/plai-scratch/jason/continual-diffusion/datasets/kcam_selected/train/20140912_115117_694_cropped/encoded_video"
    # output_video_folder = "/ubc/cs/research/plai-scratch/jason/continual-diffusion/datasets/kcam_selected/train/20140912_115117_694_cropped/decoded_video"
    dataset_path = "/ubc/cs/research/plai-scratch/jason/continual-diffusion/datasets/car"
    output_video_folder = "/ubc/cs/research/plai-scratch/jason/continual-diffusion/datasets/car"
    use_fp16 = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FINAL_FRAME_SIZE = (512, 512)

    # dataset = ChunkedCarDataset(dataset_path, window_length=1250)
    dataset = SpacedCarDataset(16, dataset_path, window_length=1250)
    dataset.set_test()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2) #, collate_fn=ContinuousCarDataset.collate_fn)

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16 if use_fp16 else torch.float32)
    vae = vae.to(device)
    vae.eval()

    Path(output_video_folder).mkdir(parents=True, exist_ok=True)

    def decode_latents(vae, latents):
        latents = latents.half() if use_fp16 else latents
        latents = latents / 0.13025
        with torch.no_grad():
            imgs = vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def save_video(frames, output_path, fps):
        height, width, _ = frames[0].shape
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), framerate=fps)
            .output(temp_video_path, pix_fmt='yuv420p', vcodec='libx264', crf=23, r=fps)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        for frame in frames:
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            process.stdin.write(frame.astype(np.uint8).tobytes())

        process.stdin.close()
        process.wait()
        os.rename(temp_video_path, output_path)

    num_batches_to_test = 10
    for i, (frames, absolute_index_map) in enumerate(dataloader):
        if i >= len(dataloader) or i >= num_batches_to_test:
            break

        print("Frames Shape:", frames.shape)
        print("Absolute Index Map:", absolute_index_map.shape)

        for b in range(frames.shape[0]):
            fps = dataset.output_fps
            video_encodings = frames[b].to(device)

            output_frames = []
            start = time.time()
            num_frames = video_encodings.shape[0]
            for frame_idx in range(num_frames):
                frame_encoding = video_encodings[frame_idx].unsqueeze(0)
                frame_encoding = frame_encoding.half() if use_fp16 else frame_encoding

                with torch.no_grad():
                    frame_img = decode_latents(vae, frame_encoding)

                frame_img = frame_img.cpu().squeeze(0).numpy()
                frame_img = np.transpose(frame_img, (1, 2, 0))
                frame_img = (frame_img * 255).astype(np.uint8)
                frame_img_resized = cv2.resize(frame_img, FINAL_FRAME_SIZE)
                output_frames.append(frame_img_resized)

            output_video_path = Path(output_video_folder) / f"video_new_test_{i}_{b}.mp4"
            save_video(output_frames, str(output_video_path), fps)

            print(f"Time Elapsed: {time.time()-start:.5f}")

            del video_encodings
            torch.cuda.empty_cache()

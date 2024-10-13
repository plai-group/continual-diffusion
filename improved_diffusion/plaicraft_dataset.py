import sqlite3
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import math
import cv2
import numpy as np
import ffmpeg
import os


"""
Sample command for zipping .db and encoded_video subdirectories.

find plaicraft_test/ -type f \( -name '*.db' \) -o -type d \( -name encoded_video  \) | zip -r plaicraft_debug.zip -@
"""

class ContinuousPlaicraftDataset(Dataset):
    # Fixed constants
    LATENT_FPS = 10  # FPS of the encoded latents
    LATENTS_PER_BATCH = 100  # Each .pt file stores 100 latents (10 seconds of data at 10 fps)
    USE_FP16 = True

    def __init__(self, dataset_path, player_names_train, player_names_test, window_length=100, output_fps=10):
        """
        Initialize the dataset with parameters.
        :param dataset_path: Path to the dataset folder.
        :param player_names: List of player names to retrieve data for.
        :param window_length: Number of latent frames in each data window.
        :param output_fps: Desired latents per second for output (default: 10, subsampled from stored latents).
        """
        self.dataset_path = Path(dataset_path)
        self.window_length = self.T = window_length  # Window length in number of latent frames
        self.global_db_path = self.dataset_path / "global_database.db"
        self.player_names_train = player_names_train
        self.player_names_test = player_names_test
        self.output_fps = output_fps
        self.sessions = None  # Will be initialized in __getitem__
        self.is_test = False

        # Validate parameters
        self._validate_parameters()

        # Initialize sessions and frame indices
        self._initialize_sessions()

    def _validate_parameters(self):
        """Ensure that input parameters are valid and within expected ranges."""
        assert isinstance(self.window_length, int) and self.window_length > 0, f"window_length must be a positive integer, but got {self.window_length}."
        assert isinstance(self.output_fps, int) and self.output_fps > 0, f"output_fps must be a positive integer, but got {self.output_fps}."

        # Check if output_fps is valid (max is 10 fps, minimum should be 1 fps)
        assert 1 <= self.output_fps <= self.LATENT_FPS, f"output_fps must be between 1 and {self.LATENT_FPS}, but got {self.output_fps}."

    def _open_connection(self):
        """Open a connection to the global database for session metadata."""
        self.connection = sqlite3.connect(self.global_db_path)

    def _get_sessions(self, player_names):
        """Retrieve all sessions belonging to the specified players that have the 'video' modality."""
        if not hasattr(self, 'connection'):
            self._open_connection()

        cur = self.connection.cursor()

        # Get sessions for each player where video is usable
        sessions = []
        for player_name in player_names:
            cur.execute("""
                SELECT session_id, start_time, frame_count, fps, player_email
                FROM session_metadata
                WHERE player_name=? AND video=1
                ORDER BY start_time ASC
            """, (player_name,))

            player_sessions = cur.fetchall()
            if player_sessions:
                sessions.append((player_name, player_sessions))

        return sessions

    def _initialize_sessions(self):
        """Initialize sessions and compute cumulative frame counts."""
        self.session_info_list = []
        self.session_boundaries = []
        total_frames = 0

        sessions = self._get_sessions(self.player_names_test if self.is_test else self.player_names_train)
        for player_name, player_sessions in sessions:
            for session in player_sessions:
                session_id, start_time, frame_count, fps, player_email = session
                latent_frames = frame_count // 3  # Latent frames at 10 fps

                session_start_frame = total_frames
                session_end_frame = total_frames + latent_frames

                session_info = {
                    'player_name': player_name,
                    'session_id': session_id,
                    'player_email': player_email,
                    'start_time': start_time,
                    'frame_count': latent_frames,
                    'session_start_frame': session_start_frame,
                    'session_end_frame': session_end_frame,
                    'paths': {
                        'video_encodings': self.dataset_path / player_email / session_id / "encoded_video"
                    }
                }
                self.session_info_list.append(session_info)
                self.session_boundaries.append((session_start_frame, session_end_frame, session_info))

                total_frames = session_end_frame
        self.total_frames = total_frames

    def _dequantize_from_int8(self, quantized_tensor, min_val, scale):
        """
        Dequantize latents from int8 to float32, broadcasting min_val and scale correctly.
        """
        # Ensure min_val and scale are broadcastable to quantized_tensor's shape
        while min_val.dim() < quantized_tensor.dim():
            min_val = min_val.unsqueeze(-1)  # Expand dimensions
        while scale.dim() < quantized_tensor.dim():
            scale = scale.unsqueeze(-1)  # Expand dimensions

        # Dequantize the latents
        dequantized = quantized_tensor.to(torch.float32) * scale + min_val
        return dequantized.half() if self.USE_FP16 else dequantized

    def __len__(self):
        """Return the total number of windows across all sessions."""
        return self.total_frames - self.window_length + 1

    def __getitem__(self, idx):
        """
        Retrieve a data item by index.
        """
        start_frame = self._get_start_frame_index(idx)
        end_frame = start_frame + self.window_length

        if end_frame > self.total_frames:
            # This should not happen as we discarded the last incomplete window
            raise IndexError(f"Index {idx} out of range.")

        frames = []
        session_segments = []
        window_frame_idx = 0  # Index within the window

        current_frame = start_frame
        frames_needed = self.window_length

        # Iterate over session boundaries to collect frames
        for session_start, session_end, session_info in self.session_boundaries:
            if current_frame >= session_end:
                continue  # Current frame is beyond this session
            elif current_frame < session_start:
                continue  # Current frame is before this session

            # Calculate the number of frames to load from this session
            session_frame_start = max(current_frame, session_start)
            session_frame_end = min(end_frame, session_end)
            num_frames = session_frame_end - session_frame_start

            # Local frame indices within the session
            local_start_frame = session_frame_start - session_start
            local_end_frame = local_start_frame + num_frames

            # Load frames from the session
            session_frames = self._load_video_encodings(session_info, local_start_frame, num_frames)
            frames.append(session_frames)

            # Record session segment information
            session_segment = {
                'player_name': session_info['player_name'],
                'session_id': session_info['session_id'],
                'window_start_idx': window_frame_idx,
                'window_end_idx': window_frame_idx + num_frames,
                'session_start_idx': local_start_frame,
                'session_end_idx': local_end_frame
            }
            session_segments.append(session_segment)

            window_frame_idx += num_frames
            current_frame += num_frames
            frames_needed -= num_frames

            if frames_needed <= 0:
                break

        # Now assemble the frames
        frames = torch.cat(frames, dim=0)
        absolute_index_map = torch.arange(start_frame, end_frame)

        # Ensure we have the correct number of frames
        if frames.shape[0] < self.window_length:
            # This should only happen at the very end, which we discard
            raise IndexError(f"Incomplete window at index {idx}, should have been discarded.")

        # Collect unique player names and session IDs
        player_names = list({segment['player_name'] for segment in session_segments})
        session_ids = list({segment['session_id'] for segment in session_segments})

        return frames, absolute_index_map

    def _get_start_frame_index(self, idx):
        return idx

    def _load_video_encodings(self, session_info, start_frame, num_frames):
        """
        Load the video latents for a specific window based on start frame and number of frames.
        This function handles loading the required latents and dequantizing them.
        """
        encodings_dir = session_info['paths']['video_encodings']
        latent_frame_count = session_info['frame_count']  # Already latent frames

        end_frame = start_frame + num_frames

        if end_frame > latent_frame_count:
            end_frame = latent_frame_count  # Adjust end_frame if it exceeds the session's frame count

        batch_idx_start = start_frame // self.LATENTS_PER_BATCH
        batch_idx_end = (end_frame - 1) // self.LATENTS_PER_BATCH  # Adjust for zero-based indexing

        accumulated_latents = []
        accumulated_min_vals = []
        accumulated_scales = []

        for batch_idx in range(batch_idx_start, batch_idx_end + 1):
            batch_file = encodings_dir / f"batch_{batch_idx:04d}.pt"
            if not batch_file.exists():
                raise FileNotFoundError(f"Latent file {batch_file} not found for session {session_info['session_id']}")

            batch_data = torch.load(batch_file)
            quantized_latents = batch_data['quantized_latents']
            min_vals = batch_data['min_vals']
            scales = batch_data['scales']

            accumulated_latents.append(quantized_latents)
            accumulated_min_vals.append(min_vals)
            accumulated_scales.append(scales)

        # Concatenate all latents, min_vals, and scales
        accumulated_latents = torch.cat(accumulated_latents, dim=0)
        accumulated_min_vals = torch.cat(accumulated_min_vals, dim=0)
        accumulated_scales = torch.cat(accumulated_scales, dim=0)

        # Calculate the global indices within the accumulated arrays
        global_start_idx = start_frame - batch_idx_start * self.LATENTS_PER_BATCH
        global_end_idx = global_start_idx + num_frames

        sliced_latents = accumulated_latents[global_start_idx:global_end_idx]
        sliced_min_vals = accumulated_min_vals[global_start_idx:global_end_idx]
        sliced_scales = accumulated_scales[global_start_idx:global_end_idx]

        # Dequantize latents (but don't decode them)
        dequantized_latents = self._dequantize_from_int8(sliced_latents, sliced_min_vals, sliced_scales)

        return dequantized_latents

    @staticmethod
    def collate_fn(batch):
        """Combine multiple data items into a batch."""
        return {
            "frames": torch.stack([item["frames"] for item in batch]),
            "player_names": [item["player_names"] for item in batch],
            "session_ids": [item["session_ids"] for item in batch],
            "session_segments": [item["session_segments"] for item in batch],
        }

    def __del__(self):
        """Close the database connection when the object is deleted."""
        if hasattr(self, 'connection'):
            self.connection.close()

    def set_train(self):
        self.is_test = False
        self._initialize_sessions()
        print('setting train mode')

    def set_test(self):
        self.is_test = True
        self._initialize_sessions()
        print('setting test mode')


class ChunkedPlaicraftDataset(ContinuousPlaicraftDataset):
    def _get_start_frame_index(self, idx):
        return idx * self.window_length

    def __len__(self):
        """Return the total number of windows across all sessions."""
        return self.total_frames // self.window_length


class SpacedPlaicraftDataset(ContinuousPlaicraftDataset):
    def __init__(self, n_data: int, *args, **kwargs):
        self.n_data = n_data
        super().__init__(*args, **kwargs)

    def _get_start_frame_index(self, idx):
        return idx * self.spacing

    def __len__(self):
        """Return the total number of windows across all sessions."""
        return self.n_data

    def _initialize_sessions(self):
        super()._initialize_sessions()
        self.spacing = self.total_frames // self.n_data
        self.spacing = self.spacing - self.spacing % self.window_length



if __name__ == "__main__":
    # Set up paths and parameters for testing
    # dataset_path = "/ubc/cs/research/plai-scratch/plaicraft/data/test_data/processed"
    # output_video_folder = "/ubc/cs/research/plai-scratch/plaicraft/data/test_data/processed/decoded_videos"
    dataset_path = "/ubc/cs/research/plai-scratch/plaicraft/data/processed"
    output_video_folder = "tmp"
    player_names = ["Kyrie"]
    use_fp16 = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FINAL_FRAME_SIZE = (1280, 768)  # (width, height)

    # Create the dataset and dataloader
    dataset = PlaicraftDataset(dataset_path, player_names, window_length=10000)
    frame_interval = dataset.frame_interval
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    # Load the AutoencoderKL model for decoding
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16 if use_fp16 else torch.float32)
    vae = vae.to(device)
    vae.eval()  # Set to evaluation mode

    # Create output directory for decoded videos
    Path(output_video_folder).mkdir(parents=True, exist_ok=True)

    def decode_latents(vae, latents):
        latents = latents.half() if use_fp16 else latents
        latents = latents / 0.13025  # Scaling factor from sample code
        with torch.no_grad():
            imgs = vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def save_video(frames, output_path, fps):
        height, width, _ = frames[0].shape
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')

        # Use ffmpeg to encode the video frames with H.264 NVENC codec
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), framerate=fps)
            .output(
                temp_video_path,
                pix_fmt='yuv420p',
                # vcodec='h264_nvenc',  # Use NVENC hardware encoder
                # preset='fast',        # NVENC encoding preset
                vcodec='libx264',
                crf=23,
                r=fps
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        for frame in frames:
            # Ensure frame is in RGB format
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            process.stdin.write(frame.astype(np.uint8).tobytes())

        process.stdin.close()
        process.wait()

        os.rename(temp_video_path, output_path)

        print(f"Saved video to {output_path}")

    num_batches_to_test = 10
    for i, batch in enumerate(dataloader):
        print("Player Names:", batch["player_name"])
        print("Session IDs:", batch["session_id"])
        print("Session Start Times:", batch["session_start_time"])
        print("Window Start Times:", batch["window_start_time"])
        print("Video Encodings Shape:", batch["video_encodings"].shape)
        print("--------------------------------------")

        # Loop through each element in the batch
        for b in range(batch["video_encodings"].shape[0]):
            player_name = batch["player_name"][b]
            session_id = batch["session_id"][b]
            window_start_time = batch["window_start_time"][b]
            fps = 10
            video_encodings = batch["video_encodings"][b].to(device)  # Move individual video encoding to GPU

            # Collect decoded frames
            output_frames = []

            # Loop through each frame in the video encoding and decode it
            start = time.time()
            num_frames = video_encodings.shape[0]
            for frame_idx in range(num_frames):
                frame_encoding = video_encodings[frame_idx].unsqueeze(0)

                frame_encoding = frame_encoding.half() if use_fp16 else frame_encoding
                # Decode the frame latent
                with torch.no_grad():
                    frame_img = decode_latents(vae, frame_encoding)

                frame_img = frame_img.cpu().squeeze(0).numpy()
                frame_img = np.transpose(frame_img, (1, 2, 0))  # Convert to (H, W, C)
                frame_img = (frame_img * 255).astype(np.uint8)

                # Resize the frame to the desired size
                frame_img_resized = cv2.resize(frame_img, FINAL_FRAME_SIZE)

                output_frames.append(frame_img_resized)

            # Define video output filename
            output_video_path = Path(output_video_folder) / f"{player_name}_{session_id}_{window_start_time}.mp4"

            # Save video using ffmpeg
            save_video(output_frames, str(output_video_path), fps)

            print(f"Time Elapsed: {time.time()-start:.5f}")

            # Free up GPU memory
            del video_encodings
            torch.cuda.empty_cache()

        if i + 1 == num_batches_to_test:
            break

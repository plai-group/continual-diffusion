import sqlite3
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import math
import time
import cv2
import numpy as np
from diffusers import AutoencoderKL
import ffmpeg
import os


class PlaicraftDataset(Dataset):
    LATENT_FPS = 10  # FPS of the encoded latents
    LATENTS_PER_BATCH = 100  # Each .pt file stores 100 latents (10 seconds of data at 10 fps)
    USE_FP16 = True

    def __init__(self, dataset_path, player_names=["Kyrie"], window_length=1000, output_fps=10):
        """
        Initialize the dataset with parameters.
        :param dataset_path: Path to the dataset folder.
        :param player_names: List of player names to retrieve data for.
        :param window_length: Length of each data window in milliseconds.
        :param output_fps: Desired latents per second for output (default: 10, subsampled from stored latents).
        """
        self.dataset_path = Path(dataset_path)
        self.window_length = window_length  # Window length in ms
        self.global_db_path = self.dataset_path / "global_database.db"
        self.player_names = player_names
        self.output_fps = output_fps
        self.sessions = None  # Will be initialized in __getitem__

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        assert isinstance(self.window_length, int) and self.window_length > 0, f"window_length must be a positive integer, but got {self.window_length}."
        assert isinstance(self.output_fps, int) and self.output_fps > 0, f"output_fps must be a positive integer, but got {self.output_fps}."

        """Ensure that input parameters are valid and within expected ranges."""
        # Check if output_fps is valid (max is 10 fps, minimum should be 1 fps)
        assert 1 <= self.output_fps <= self.LATENT_FPS, f"output_fps must be between 1 and {self.LATENT_FPS}, but got {self.output_fps}."

        # Check if window_length is valid (must allow at least one frame)
        min_window_length = 1000 / self.output_fps  # Minimum window length in ms for 1 frame
        assert self.window_length >= min_window_length, f"window_length must be at least {math.ceil(min_window_length)} ms for the chosen output_fps of {self.output_fps}, but got {self.window_length}."

    def _open_connection(self):
        """Open a connection to the global database for session metadata."""
        self.connection = sqlite3.connect(self.global_db_path)

    def _get_sessions(self):
        """Retrieve all sessions belonging to the specified players that have the 'video' modality."""
        if not hasattr(self, 'connection'):
            self._open_connection()

        cur = self.connection.cursor()

        # Get sessions for each player where video is usable
        sessions = []
        for player_name in self.player_names:
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

    def _get_player_session_index(self, idx):
        """Map the global index to the corresponding player and session."""
        count = 0
        for player_idx, (_, sessions) in enumerate(self.sessions):
            session_len = sum(math.ceil((frame_count // 3) /
                                        (self.window_length * self.LATENT_FPS // (1000)))
                              for session_id, start_time, frame_count, fps, db_path in sessions)
            if idx < count + session_len:
                return player_idx, idx - count
            count += session_len
        raise IndexError(f"Index {idx} out of range.")

    def _load_session_info(self, session_id, db_path):
        """Load session metadata from the individual session database."""
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("""
            SELECT session_id, video_usable, fps, frame_count, start_time
            FROM session WHERE session_id = ?
        """, (session_id,))
        session_info = cur.fetchone()
        con.close()

        if not session_info:
            raise ValueError(f"No session info found for session ID: {session_id}")

        session_folder = Path(db_path).parent
        return {
            "session_id": session_info[0],
            "fps": session_info[2],  # This is the original 30 fps
            "frame_count": session_info[3],  # This is the original 30 fps frame count
            "start_time": session_info[4],
            "paths": {
                "video_encodings": session_folder / "encoded_video"  # Latents at 10 fps
            }
        }

    def _load_video_encodings(self, session_info, relative_start_time):
        """
        Load the video latents for a specific time window based on relative start time.
        This function handles loading the required latents and dequantizing them.
        """
        encodings_dir = session_info['paths']['video_encodings']
        latent_frame_count = session_info['frame_count'] // 3  # Downsampled to 10 fps

        start_frame = int(relative_start_time / 1000 * self.output_fps)  # Latent start frame
        end_frame = start_frame + int(self.window_length / 1000 * self.output_fps)  # Latent end frame

        max_frame = latent_frame_count
        if end_frame > max_frame:
            end_frame = max_frame

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

        # Calculate the global indices
        global_start_idx = start_frame - batch_idx_start * self.LATENTS_PER_BATCH
        global_end_idx = global_start_idx + (end_frame - start_frame)

        sliced_latents = accumulated_latents[global_start_idx:global_end_idx]
        sliced_min_vals = accumulated_min_vals[global_start_idx:global_end_idx]
        sliced_scales = accumulated_scales[global_start_idx:global_end_idx]

        # Dequantize latents (but don't decode them)
        dequantized_latents = self._dequantize_from_int8(sliced_latents, sliced_min_vals, sliced_scales)

        # Handle padding if the window length exceeds the available frames
        expected_length = int(self.window_length / 1000 * self.output_fps)
        current_length = dequantized_latents.shape[0]
        if current_length < expected_length:
            padding_size = expected_length - current_length
            pad_shape = (padding_size,) + dequantized_latents.shape[1:]
            padding = torch.zeros(pad_shape, dtype=dequantized_latents.dtype)
            dequantized_latents = torch.cat([dequantized_latents, padding], dim=0)

        return dequantized_latents

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
        if self.sessions is None:
            self.sessions = self._get_sessions()

        total_windows = 0

        for _, sessions in self.sessions:
            for session_id, start_time, frame_count, fps, db_path in sessions:
                # Total number of latent frames processed (frame_count is for 30 fps)
                latent_frames = frame_count // 3  # Downsampled to 10 fps

                # Each frame generates latents, calculate the number of windows based on window length
                latents_per_window = self.window_length * self.LATENT_FPS // 1000
                num_windows = math.ceil(latent_frames / latents_per_window)

                total_windows += num_windows

        return total_windows

    def __getitem__(self, idx):
        """
        Retrieve a data item by index, loading the corresponding player and session data,
        then loading the video encodings for the relevant window.
        """
        if self.sessions is None:
            self.sessions = self._get_sessions()

        player_idx, session_offset = self._get_player_session_index(idx)
        player_name, session_data = self.sessions[player_idx]

        # Map the offset to the correct session
        session_cumulative_idx = 0
        for session_idx, session in enumerate(session_data):
            session_id, session_start_time, frame_count, fps, player_email = session
            latent_frames = frame_count // 3  # Latent frames at 10 fps
            latents_per_window = self.window_length * self.LATENT_FPS // 1000
            num_windows = math.ceil(latent_frames / latents_per_window)
            if session_cumulative_idx + num_windows > session_offset:
                # This is the correct session
                window_idx = session_offset - session_cumulative_idx
                break
            session_cumulative_idx += num_windows

        db_path = self.dataset_path / player_email / session_id / f"{session_id}.db"

        # Load session info
        session_info = self._load_session_info(session_id, db_path)

        # Calculate relative start time for the current window (in ms)
        relative_start_time = window_idx * self.window_length

        # Load video encodings for the current window using the relative start time
        video_encodings = self._load_video_encodings(session_info, relative_start_time)
        return video_encodings, {}

        # return {
        #     "player_name": player_name,
        #     "session_id": session_id,
        #     "session_start_time": session_start_time,  # Absolute session start time
        #     "window_start_time": relative_start_time,  # Relative time with respect to the start of the video
        #     "video_encodings": video_encodings,
        # }

    @staticmethod
    def collate_fn(batch):
        """Combine multiple data items into a batch."""
        return {
            "player_name": [item["player_name"] for item in batch],
            "session_id": [item["session_id"] for item in batch],
            "session_start_time": [item["session_start_time"] for item in batch],
            "window_start_time": [item["window_start_time"] for item in batch],
            "video_encodings": torch.stack([item["video_encodings"] for item in batch])  # Stack the tensors directly
        }

    def __del__(self):
        """Close the database connection when the object is deleted."""
        if hasattr(self, 'connection'):
            self.connection.close()

    def set_test(self):
        # FIXME: IMPLEMENT TEST SET LOGIC HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pass



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

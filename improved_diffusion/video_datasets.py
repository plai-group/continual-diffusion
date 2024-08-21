from abc import abstractmethod
import json
import os
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from pathlib import Path
import shutil
from mpi4py import MPI

from .test_util import Protect



video_data_paths_dict = {
    "ball_stn":            "datasets/ball_stn",
    "ball_nstn":           "datasets/ball_nstn",
    "streaming_ball_stn":  "datasets/ball_stn",
    "streaming_ball_nstn": "datasets/ball_nstn",
    "wmaze":               "datasets/windows_maze",
    "streaming_wmaze":     "datasets/windows_maze",
    "mine":                "datasets/continual_minecraft",
    "streaming_mine":      "datasets/continual_minecraft",
    "minerl":              "datasets/minerl_navigate-torch",
    "mazes_cwvae":         "datasets/gqn_mazes-torch",
    "carla_no_traffic":    "datasets/carla/no-traffic",
    "carla_no_traffic_2x": "datasets/carla/no-traffic",
    "carla_no_traffic_2x_encoded": "datasets/carla/no-traffic-encoded",
}

default_T_dict = {
    "ball_stn":            10,
    "ball_nstn":           10,
    "streaming_ball_stn":  10,  # gets reset to 1 for the dataset
    "streaming_ball_nstn": 10,  # gets reset to 1 for the dataset
    "wmaze":               20,
    "streaming_wmaze":     20,
    "mine":                500,
    "streaming_mine":      1,
    "minerl":              500,
    "mazes_cwvae":         300,
    "carla_no_traffic":    1000,
    "carla_no_traffic_2x": 1000,
    "carla_no_traffic_2x_encoded": 1000,
}

default_image_size_dict = {
    "ball_stn":            32,
    "ball_nstn":           32,
    "streaming_ball_stn":  32,
    "streaming_ball_nstn": 32,
    "wmaze":               64,
    "streaming_wmaze":     64,
    "mine":                64,
    "streaming_mine":      64,
    "minerl":              64,
    "mazes_cwvae":         64,
    "carla_no_traffic":    128,
    "carla_no_traffic_2x": 256,
    "carla_no_traffic_2x_encoded": 32,
}

data_encoding_stats_dict = {
    "carla_no_traffic_2x_encoded": "datasets/carla/no-traffic-encoded/encoded_train_norm_stats.pt",
}


def load_data(dataset_name, batch_size, T=None, deterministic=False, num_workers=1, return_dataset=False, resume_id='', restart_index=None, seed=0):
    data_path = video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    shard = MPI.COMM_WORLD.Get_rank()
    num_shards = MPI.COMM_WORLD.Get_size()
    if dataset_name.startswith("streaming"):
        T, deterministic = 1, True
    if "ball_stn" in dataset_name:
        dataset = ContinuousBaseDataset(data_path, T=T, seed=seed, restart_index=restart_index)
    elif "ball_nstn" in dataset_name:
        dataset = ContinuousBaseDataset(data_path, T=T, seed=seed)
    elif "wmaze" in dataset_name:
        dataset = ContinuousBaseDataset(data_path, T=T, seed=seed)
    elif dataset_name == "mine":
        dataset = MineDataset(data_path, shard=shard, num_shards=num_shards, T=T)
    elif dataset_name == "streaming_mine":
        dataset = MineDataset(data_path, shard=shard, num_shards=num_shards, T=1)
        deterministic = True
    elif dataset_name == "minerl":
        data_path = os.path.join(data_path, "train")
        dataset = MineRLDataset(data_path, shard=shard, num_shards=num_shards, T=T)
    elif dataset_name == "mazes_cwvae":
        data_path = os.path.join(data_path, "train")
        dataset = GQNMazesDataset(data_path, shard=shard, num_shards=num_shards, T=T)
    elif dataset_name == "carla_no_traffic":
        dataset = CarlaDataset(train=True, path=data_path, shard=shard, num_shards=num_shards, T=T)
    elif dataset_name == "carla_no_traffic_2x":
        dataset = Carla2xDataset(train=True, path=data_path, shard=shard, num_shards=num_shards, T=T)
    elif dataset_name == "carla_no_traffic_2x_encoded":
        dataset = Carla2xDataset(train=True, path=data_path, shard=shard, num_shards=num_shards, T=T, encoded=True)
    else:
        raise Exception("no dataset", dataset_name)

    if resume_id and deterministic:  # start from specific data stream index.
        with open(os.path.join('checkpoints', resume_id, 'replay_state.json')) as f:
            n_observed = json.load(f)['n_obs']
            start_index = n_observed // dataset.T + 1
            print(f"starting deterministic dataloader from {n_observed+1}-th datapoint.")
        dataset = th.utils.data.Subset(dataset, indices=range(start_index, len(dataset)))

    if return_dataset:
        return dataset
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(not deterministic), num_workers=num_workers, drop_last=True
        )
        counter, deterministic_epoch_limit = 0, 1
        while True:
            yield from loader

            counter += 1
            if deterministic and counter >= deterministic_epoch_limit:
                raise StopIteration()


def get_train_dataset(dataset_name, T=None, seed=0):
    return load_data(
        dataset_name, return_dataset=False, T=T, batch_size=None, deterministic=False, num_workers=None, seed=0
    )


def get_test_dataset(dataset_name, T=None, seed=0, n_data=None):
    if dataset_name == "mazes":
        raise Exception('Deprecated dataset.')
    data_root = Path(os.environ["DATA_ROOT"]  if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "" else ".")
    data_path = data_root / video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    if "ball_stn" in dataset_name:
        dataset = ChunkedBaseDataset(data_path, T=T, seed=seed)
    elif "ball_nstn" in dataset_name:
        dataset = SpacedBaseDataset(n_data, data_path, T=T, seed=seed)
    elif "wmaze" in dataset_name:
        dataset = ChunkedBaseDataset(data_path, T=T, seed=seed)
    elif dataset_name == "mine":
        dataset = MineDataset(data_path, shard=0, num_shards=1, T=T)
    elif dataset_name == "streaming_mine":
        dataset = MineDataset(data_path, shard=0, num_shards=0, T=T)
    elif dataset_name == "minerl":
        data_path = os.path.join(data_path, "test")
        dataset = MineRLDataset(data_path, shard=0, num_shards=1, T=T)
    elif dataset_name == "mazes_cwvae":
        data_path = os.path.join(data_path, "test")
        dataset = GQNMazesDataset(data_path, shard=0, num_shards=1, T=T)
    elif dataset_name == "carla_no_traffic":
        dataset = CarlaDataset(train=False, path=data_path, shard=0, num_shards=1, T=T)
    elif dataset_name == "carla_no_traffic_2x":
        dataset = Carla2xDataset(train=False, path=data_path, shard=0, num_shards=1, T=T)
    elif dataset_name == "carla_no_traffic_2x_encoded":
        dataset = Carla2xDataset(train=False, path=data_path, shard=0, num_shards=1, T=T, encoded=True)
    else:
        raise Exception("no dataset", dataset_name)
    dataset.set_test()
    return dataset


def get_vis_dataset(dataset_name, T=None, seed=0):
    if dataset_name == "mazes":
        raise Exception('Deprecated dataset.')
    data_root = Path(os.environ["DATA_ROOT"]  if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "" else ".")
    data_path = data_root / video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    if "ball_stn" in dataset_name:
        dataset = ChunkedBaseDataset(data_path, T=T, seed=seed)
    elif "ball_nstn" in dataset_name:
        dataset = ChunkedBaseDataset(data_path, T=T, seed=seed)
    elif "wmaze" in dataset_name:
        dataset = ChunkedBaseDataset(data_path, T=T, seed=seed)
    else:
        raise Exception("no dataset", dataset_name)
    dataset.set_test()
    return dataset


class BaseDataset(Dataset):
    """ The base class for our video datasets. It is used for datasets where each video is stored under <dataset_root_path>/<split>
        as a single file. This class provides the ability of caching the dataset items in a temporary directory (if
        specified as an environment variable DATA_ROOT) as the items are read. In other words, every time an item is
        retrieved from the dataset, it will try to load it from the temporary directory first. If it is not found, it
        will be first copied from the original location.

        This class provides a default implementation for __len__ as the number of file in the dataset's original directory.
        It also provides the following two helper functions:
        - cache_file: Given a path to a dataset file, makes sure the file is copied to the temporary directory. Does
        nothing unless DATA_ROOT is set.
        - get_video_subsequence: Takes a video and a video length as input. If the video length is smaller than the
          input video's length, it returns a random subsequence of the video. Otherwise, it returns the whole video.
        A child class should implement the following methods:
        - getitem_path: Given an index, returns the path to the video file.
        - loaditem: Given a path to a video file, loads and returns the video.
        - postprocess_video: Given a video, performs any postprocessing on the video.

    Args:
        path (str): path to the dataset split
    """
    def __init__(self, path, T):
        super().__init__()
        self.T = T
        self.path = Path(path)
        self.is_test = False

    def __len__(self):
        path = self.get_src_path(self.path)
        return len(list(path.iterdir()))

    def __getitem__(self, idx):
        path = self.getitem_path(idx)
        self.cache_file(path)
        try:
            video = self.loaditem(path)
        except Exception as e:
            print(f"Failed on loading {path}")
            raise e
        video = self.postprocess_video(video)
        return self.get_video_subsequence(video, self.T), {}

    @abstractmethod
    def getitem_path(self, idx):
        raise NotImplementedError()

    @abstractmethod
    def loaditem(self, path):
        raise NotImplementedError()

    @abstractmethod
    def postprocess_video(self, video):
        raise NotImplementedError()

    def cache_file(self, path):
        # Given a path to a dataset item, makes sure that the item is cached in the temporary directory.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            src_path = self.get_src_path(path)
            with Protect(path):
                shutil.copyfile(str(src_path), str(path))

    @staticmethod
    def get_src_path(path):
        """ Returns the source path to a file. This function is mainly used to handle SLURM_TMPDIR on ComputeCanada.
            If DATA_ROOT is defined as an environment variable, the datasets are copied to it as they are accessed. This function is called
            when we need the source path from a given path under DATA_ROOT.
        """
        if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "":
            # Verify that the path is under
            data_root = Path(os.environ["DATA_ROOT"])
            assert data_root in path.parents, f"Expected dataset item path ({path}) to be located under the data root ({data_root})."
            src_path = Path(*path.parts[len(data_root.parts):]) # drops the data_root part from the path, to get the relative path to the source file.
            return src_path
        return path

    def set_test(self):
        self.is_test = True
        print('setting test mode')

    def get_video_subsequence(self, video, T):
        if T is None:
            return video
        if T < len(video):
            # Take a subsequence of the video.
            start_i = 0 if self.is_test else np.random.randint(len(video) - T + 1)
            video = video[start_i:start_i+T]
        assert len(video) == T
        return video


class ContinuousBaseDataset(Dataset):
    """
    A dataset that takes one long video saved in multiple .npy files and returns a size T sliding window of
    video frames indexed by the location of the sliding window's first frame in the video.

    __getitem__ returns data of shape <1 x T x ...>
    self.chunk_size denotes the number of frames present in each npy file.
    T denotes the number of frames that the dataset should return to the model per item.
    """
    def __init__(self, path, T=1, seed=0, restart_index=None):
        super().__init__()
        self.T = T
        self.path = Path(path)
        self.is_test = False
        self.restart_index = int(restart_index) if restart_index is not None else None

        config = json.load(open(self.path / 'config.json'))
        self.T_total = config['T_total']
        self.chunk_size = config['chunk_size']
        assert self.T_total % self.T == 0
        assert self.T_total % self.chunk_size == 0
        assert self.chunk_size % self.T == 0
        assert self.restart_index is None or (self.restart_index > 0 and self.restart_index % self.T == 0)

        self.train_path = self.path / 'train' / str(seed)
        self.test_path = self.path / 'test' / str(seed)

    def __len__(self):
        return self.T_total - self.T + 1

    def __getitem__(self, idx):
        if self.restart_index is not None:
            idx = idx % self.restart_index

        paths = self.getitem_paths(idx)
        self.cache_files(paths)
        try:
            video = self.loaditem(paths)
        except Exception as e:
            print(f"Failed on loading {paths}")
            raise e
        video = self.get_video_subsequence(video, idx)
        return self.postprocess_video(video), {}

    def getitem_paths(self, idx):
        chunk_idxs = [idx // self.chunk_size]
        if (idx % self.chunk_size) + self.T > self.chunk_size:
            chunk_idxs.append((idx // self.chunk_size) + 1)
        return [(self.test_path if self.is_test else self.train_path) / f"{cidx}.npy" for cidx in chunk_idxs]

    def loaditem(self, paths):
        loaded = [np.load(path) for path in paths]
        return np.concatenate(loaded, axis=0)

    def postprocess_video(self, video):
        byte_to_tensor = lambda x: ToTensor()(x)
        video = th.stack([byte_to_tensor(frame).float() for frame in video])
        video = 2 * video - 1
        return video

    def cache_files(self, paths):
        # Given a path to a dataset item, makes sure that the item is cached in the temporary directory.
        for path in paths:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                src_path = self.get_src_path(path)
                with Protect(path):
                    shutil.copyfile(str(src_path), str(path))

    @staticmethod
    def get_src_path(path):
        """ Returns the source path to a file. This function is mainly used to handle SLURM_TMPDIR on ComputeCanada.
            If DATA_ROOT is defined as an environment variable, the datasets are copied to it as they are accessed. This function is called
            when we need the source path from a given path under DATA_ROOT.
        """
        if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "":
            # Verify that the path is under
            data_root = Path(os.environ["DATA_ROOT"])
            assert data_root in path.parents, f"Expected dataset item path ({path}) to be located under the data root ({data_root})."
            src_path = Path(*path.parts[len(data_root.parts):]) # drops the data_root part from the path, to get the relative path to the source file.
            return src_path
        return path

    def set_test(self):
        self.is_test = True
        print('setting test mode')

    def get_video_subsequence(self, video, idx):
        # Take a subsequence of the video.
        start_i = idx % self.chunk_size
        video = video[start_i:start_i+self.T]
        assert len(video) == self.T
        return video


class ChunkedBaseDataset(ContinuousBaseDataset):
    """
    A dataset that takes one long video saved in multiple .npy files and returns a size T video frame
    subsequence indexed by the location of the sliding window's first frame in the video divided by T.

    __getitem__ returns data of shape <1 x T x ...>
    self.chunk_size denotes the number of frames present in each npy file.
    T denotes the number of frames that the dataset should return to the model per item.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.T_total // self.T

    def getitem_paths(self, idx):
        chunk_idxs = [(idx * self.T) // self.chunk_size]
        return [(self.test_path if self.is_test else self.train_path) / f"{cidx}.npy" for cidx in chunk_idxs]

    def get_video_subsequence(self, video, idx):
        # Take a subsequence of the video.
        start_i = (idx * self.T) % self.chunk_size
        video = video[start_i:start_i+self.T]
        assert len(video) == self.T
        return video


class SpacedBaseDataset(ContinuousBaseDataset):
    """
    A dataset that takes one long video saved in multiple .npy files and returns a size T video frame
    subsequence indexed by the location of the sliding window's first frame in the video divided by T.

    __getitem__ returns data of shape <1 x T x ...>
    self.chunk_size denotes the number of frames present in each npy file.
    T denotes the number of frames that the dataset should return to the model per item.
    """
    def __init__(self, n_data: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_data = n_data
        self.spacing = self.T_total // self.n_data
        assert self.spacing % self.T == 0
        assert self.restart_index is None

    def __len__(self):
        return self.n_data

    def getitem_paths(self, idx):
        chunk_idxs = [(idx * self.spacing) // self.chunk_size]
        return [(self.test_path if self.is_test else self.train_path) / f"{cidx}.npy" for cidx in chunk_idxs]

    def get_video_subsequence(self, video, idx):
        # Take a subsequence of the video.
        start_i = (idx * self.spacing) % self.chunk_size
        video = video[start_i:start_i+self.T]
        assert len(video) == self.T
        return video


class WindowsMazeDataset(ContinuousBaseDataset):
    """
    Youtube video download and preprocessing

    yt-dlp "https://www.youtube.com/watch?v=MHGnSqr9kK8&ab_channel=Dprotp" -S res:256
    ffmpeg -i '10 Hours of Windows 3D Maze [MHGnSqr9kK8].webm' -filter:v "fps=30,crop=240:240:30:0,scale=64:64" windows_maze_10h_r64.mp4
    """
    def __init__(self, path, shard, num_shards, T, seed):
        assert shard == 0, "Distributed training is not supported by the Bouncing Ball dataset yet."
        assert num_shards == 1, "Distributed training is not supported by the Bouncing Ball dataset yet."
        super().__init__(path=path, T=T, seed=seed)


class MineDataset(ContinuousBaseDataset):
    def __init__(self, path, shard, num_shards, T, seed):
        assert shard == 0, "Distributed training is not supported by the Bouncing Ball dataset yet."
        assert num_shards == 1, "Distributed training is not supported by the Bouncing Ball dataset yet."
        super().__init__(path=path, T=T, seed=seed)


class CarlaDataset(BaseDataset):
    def __init__(self, train, path, shard, num_shards, T):
        super().__init__(path=path, T=T)
        self.split_path = self.path / f"video_{'train' if train else 'test'}.csv"
        self.cache_file(self.split_path)
        self.fnames = [line.rstrip('\n').split('/')[-1] for line in open(self.split_path, 'r').readlines() if '.pt' in line]
        self.fnames = self.fnames[shard::num_shards]
        print(f"Loading {len(self.fnames)} files (Carla dataset).")

    def loaditem(self, path):
        return th.load(path)

    def getitem_path(self, idx):
        return self.path / self.fnames[idx]

    def postprocess_video(self, video):
        return -1 + 2 * (video.permute(0, 3, 1, 2).float()/255)

    def __len__(self):
        return len(self.fnames)


class Carla2xDataset(CarlaDataset):
    def __init__(self, train, path, shard, num_shards, T, encoded=False):
        super().__init__(train, path, shard, num_shards, T)
        self.encoded = encoded
        if self.encoded:
            self.fnames = ["encoded_" + fname for fname in self.fnames]
        print(f"Loading {len(self.fnames)} files (Carla dataset).")

    def postprocess_video(self, video):
        if not self.encoded:
            result = -1 + 2 * (video.permute(0, 3, 1, 2).float()/255)
            # remove frames before upsampling to save memory if small --T is used
            video = self.get_video_subsequence(result, self.T)
            video = th.nn.functional.interpolate(result, scale_factor=2)
        return video


class GQNMazesDataset(BaseDataset):
    """ based on https://github.com/iShohei220/torch-gqn/blob/master/gqn_dataset.py .
    """
    def __init__(self, path, shard, num_shards, T):
        assert shard == 0, "Distributed training is not supported by the MineRL dataset yet."
        assert num_shards == 1, "Distributed training is not supported by the MineRL dataset yet."
        super().__init__(path=path, T=T)

    def getitem_path(self, idx):
        return self.path / f"{idx}.npy"

    def loaditem(self, path):
        return np.load(path)

    def postprocess_video(self, video):
        byte_to_tensor = lambda x: ToTensor()(x)
        video = th.stack([byte_to_tensor(frame) for frame in video])
        video = 2 * video - 1
        return video


class MineRLDataset(BaseDataset):
    def __init__(self, path, shard, num_shards, T):
        assert shard == 0, "Distributed training is not supported by the MineRL dataset yet."
        assert num_shards == 1, "Distributed training is not supported by the MineRL dataset yet."
        super().__init__(path=path, T=T)

    def getitem_path(self, idx):
        return self.path / f"{idx}.npy"

    def loaditem(self, path):
        return np.load(path)

    def postprocess_video(self, video):
        byte_to_tensor = lambda x: ToTensor()(x)
        video = th.stack([byte_to_tensor(frame) for frame in video])
        video = 2 * video - 1
        return video

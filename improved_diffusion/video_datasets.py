from abc import abstractmethod
import json
import os
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import ToTensor
import torch.distributed as dist
from pathlib import Path
import shutil
from typing import Tuple
from mpi4py import MPI
from improved_diffusion.replay_sampler import DistributedReplaySampler

from .train_util import get_blob_logdir
from .test_util import Protect
from .plaicraft_dataset import ContinuousPlaicraftDataset, SpacedPlaicraftDataset, ChunkedPlaicraftDataset



video_data_paths_dict = {
    "ball_stn":            "datasets/ball_stn",
    "ball_nstn":           "datasets/ball_nstn",
    "streaming_ball_stn":  "datasets/ball_stn",
    "streaming_ball_nstn": "datasets/ball_nstn",
    "wmaze":               "datasets/windows_maze",
    "streaming_wmaze":     "datasets/windows_maze",
    "plaicraft":           "datasets/plaicraft",
    "streaming_plaicraft": "datasets/plaicraft",
}

default_T_dict = {
    "ball_stn":            10,
    "ball_nstn":           10,
    "streaming_ball_stn":  10,  # gets reset to 1 for the dataset
    "streaming_ball_nstn": 10,  # gets reset to 1 for the dataset
    "wmaze":               20,
    "streaming_wmaze":     20,
    "plaicraft":           20,
    "streaming_plaicraft": 20,
}

default_image_size_dict = {
    "ball_stn":            32,
    "ball_nstn":           32,
    "streaming_ball_stn":  32,
    "streaming_ball_nstn": 32,
    "wmaze":               64,
    "streaming_wmaze":     64,
    "plaicraft":           160,
    "streaming_plaicraft": 160,
}

eval_dataset_configs = {"default": "default", "continuous": "continuous", "chunked": "chunked"}


def get_data_path(dataset_name):
    # If DATA_ROOT environment variable is specified, it is assumed to point at local node storage. If the data is already there,
    # the dataset objects will read off them. Otherwise, data will be copied from the shared storage as they are retrieved.
    data_path = video_data_paths_dict[dataset_name]
    if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "":
        data_root = Path(os.environ["DATA_ROOT"])
        data_path = data_root / data_path
        data_path.mkdir(parents=True, exist_ok=True)
    return data_path


def load_data(dataset_name, batch_size, T=None, deterministic=False, num_workers=1, return_dataset=False,
              resume_id='', seed=0, buffer_size=None, n_sequential=1, save_every=None):
    data_path = get_data_path(dataset_name)
    T = default_T_dict[dataset_name] if T is None else T
    shard = MPI.COMM_WORLD.Get_rank()
    num_shards = MPI.COMM_WORLD.Get_size()

    if dataset_name.startswith("streaming"):
        deterministic = True
    if "ball_stn" in dataset_name:
        dataset = ContinuousBaseDataset(data_path, T=T, seed=seed)
    elif "ball_nstn" in dataset_name:
        dataset = ContinuousBaseDataset(data_path, T=T, seed=seed)
    elif "wmaze" in dataset_name:
        dataset = ContinuousBaseDataset(data_path, T=T, seed=seed)
    elif "plaicraft" in dataset_name:
        dataset = ContinuousPlaicraftDataset(data_path, window_length=T,
                                             player_names_train=["Alex"],
                                             player_names_test=["Kyrie"])
    else:
        raise Exception("no dataset", dataset_name)

    if return_dataset:
        return dataset

    epoch = 0
    if deterministic:
        save_path = os.path.join(get_blob_logdir(resume_id), 'replay_state.pt') if dist.get_rank() == 0 else ''
        sampler = DistributedReplaySampler(dataset, batch_size, buffer_size=buffer_size, seed=seed,
                                           n_sequential=n_sequential, save_args=dict(path=save_path, every=save_every))
        if resume_id:
            load_path = os.path.join(get_blob_logdir(resume_id), 'replay_state.pt')
            sampler.load_sampler(path=load_path)
            print(f"starting replay dataloader from data index {sampler.start_index}.")
    else:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
        sampler.set_epoch(epoch)

    batch_size = batch_size // dist.get_world_size()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    while True:
        yield from loader
        epoch += 1
        if deterministic:
            raise StopIteration()
        else:
            sampler.set_epoch(epoch)

def get_eval_dataset(dataset_name, T=None, seed=0, train=False, eval_dataset_config=eval_dataset_configs["default"],
                     spacing_kwargs=dict(n_data=None, frame_range=(0, None))):
    """
    """
    data_path = get_data_path(dataset_name)
    T = default_T_dict[dataset_name] if T is None else T
    if "ball_stn" in dataset_name:
        shared_args = dict(dataset_path=data_path,  T=T, seed=seed)
        if eval_dataset_config == eval_dataset_configs["continuous"]:
            dataset = ContinuousBaseDataset(**shared_args)
        else:
            dataset = ChunkedBaseDataset(frame_range=spacing_kwargs["frame_range"], **shared_args)
    elif "ball_nstn" in dataset_name:
        shared_args = dict(dataset_path=data_path, window_length=T, seed=seed)
        if eval_dataset_config == eval_dataset_configs["continuous"]:
            dataset = ContinuousBaseDataset(**shared_args)
        elif eval_dataset_config == eval_dataset_configs["chunked"]:
            dataset = ChunkedBaseDataset(frame_range=spacing_kwargs["frame_range"], **shared_args)
        else:
            dataset = SpacedBaseDataset(**spacing_kwargs, **shared_args)
    elif "wmaze" in dataset_name:
        shared_args = dict(dataset_path=data_path,  T=T, seed=seed)
        if eval_dataset_config == eval_dataset_configs["continuous"]:
            dataset = ContinuousBaseDataset(**shared_args)
        else:
            dataset = ChunkedBaseDataset(frame_range=spacing_kwargs["frame_range"], **shared_args)
    elif "plaicraft" in dataset_name:
        shared_args = dict(dataset_path=data_path, window_length=T,
                           player_names_train=["Alex"], player_names_test=["Kyrie"])
        if eval_dataset_config == eval_dataset_configs["continuous"]:
            dataset = ContinuousPlaicraftDataset(**shared_args)
        elif eval_dataset_config == eval_dataset_configs["chunked"]:
            dataset = ChunkedPlaicraftDataset(frame_range=spacing_kwargs["frame_range"], **shared_args)
        else:
            dataset = SpacedPlaicraftDataset(**spacing_kwargs, **shared_args)
    else:
        raise Exception("no dataset", dataset_name)
    if not train:
        dataset.set_test()
    return dataset


def get_train_dataset(dataset_name, T=None, seed=0):
    return load_data(dataset_name, T=T, batch_size=None, seed=0, return_dataset=True)


def get_test_dataset(dataset_name, T=None, seed=0, n_data=None):
    data_path = get_data_path(dataset_name)
    T = default_T_dict[dataset_name] if T is None else T
    if "ball_stn" in dataset_name:
        dataset = ChunkedBaseDataset(data_path, T=T, seed=seed)
    elif "ball_nstn" in dataset_name:
        dataset = SpacedBaseDataset(n_data, data_path, T=T, seed=seed)
    elif "wmaze" in dataset_name:
        dataset = ChunkedBaseDataset(data_path, T=T, seed=seed)
    elif "plaicraft" in dataset_name:
        dataset = SpacedPlaicraftDataset(n_data, data_path, window_length=T,
                                         player_names_train=["Alex"],
                                         player_names_test=["Kyrie"])
    else:
        raise Exception("no dataset", dataset_name)
    dataset.set_test()
    return dataset


def get_vis_dataset(dataset_name, T=None, seed=0):
    data_path = get_data_path(dataset_name)
    T = default_T_dict[dataset_name] if T is None else T
    if "ball_stn" in dataset_name:
        dataset = ChunkedBaseDataset(data_path, T=T, seed=seed)
    elif "ball_nstn" in dataset_name:
        dataset = ChunkedBaseDataset(data_path, T=T, seed=seed)
    elif "wmaze" in dataset_name:
        dataset = ChunkedBaseDataset(data_path, T=T, seed=seed)
    elif "plaicraft" in dataset_name:
        dataset = ChunkedPlaicraftDataset(data_path, window_length=T,
                                          player_names_train=["Alex"],
                                          player_names_test=["Kyrie"])
    else:
        raise Exception("no dataset", dataset_name)
    dataset.set_test()
    return dataset


class ContinuousBaseDataset(Dataset):
    """
    A dataset that takes one long video saved in multiple .npy files and returns a size T sliding window of
    video frames indexed by the location of the sliding window's first frame in the video.

    __getitem__ returns data of shape <1 x T x ...>
    self.chunk_size denotes the number of frames present in each npy file.
    T denotes the number of frames that the dataset should return to the model per item.
    """
    def __init__(self, dataset_path, T=1, seed=0, restart_index=None):
        super().__init__()
        self.T = T
        self.path = Path(dataset_path)
        self.is_test = False
        self.restart_index = int(restart_index) if restart_index is not None else None

        config = self.get_config(self.path / 'config.json')
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
        frames = self.postprocess_video(video)
        absolute_index_map = th.arange(idx, idx+self.T)
        return frames, absolute_index_map

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
            with Protect(path):
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    src_path = self.get_src_path(path)
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

    @staticmethod
    def get_config(path):
        if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "":
            # Verify that the path is under
            data_root = Path(os.environ["DATA_ROOT"])
            assert data_root in path.parents, f"Expected dataset item path ({path}) to be located under the data root ({data_root})."
            path = Path(*path.parts[len(data_root.parts):]) # drops the data_root part from the path, to get the relative path to the source file.
        return json.load(open(path))

    def set_train(self):
        self.is_test = False
        print('setting train mode')

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
    def __init__(self, frame_range: Tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_range = frame_range
        if self.frame_range[1] is None:
            self.frame_range = (self.frame_range[0], self.T_total)

    def __len__(self):
        return (self.frame_range[1]-self.frame_range[0])  // self.T

    def getitem_paths(self, idx):
        chunk_idxs = [(self.frame_range[0] + idx * self.T) // self.chunk_size]
        return [(self.test_path if self.is_test else self.train_path) / f"{cidx}.npy" for cidx in chunk_idxs]

    def get_video_subsequence(self, video, idx):
        # Take a subsequence of the video.
        start_i = (self.frame_range[0] + idx * self.T) % self.chunk_size
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
    def __init__(self, n_data: int, frame_range: Tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_data = n_data
        self.frame_range = frame_range
        if self.frame_range[1] is None:
            self.frame_range = (self.frame_range[0], self.T_total)

        self.spacing = (self.frame_range[1]-self.frame_range[0]) // self.n_data
        assert self.spacing % self.T == 0
        assert 0<=self.frame_range[0] and self.frame_range[0]+self.T<self.frame_range[1] and self.frame_range[1]<=self.n_data
        assert self.restart_index is None

    def __len__(self):
        return self.n_data

    def getitem_paths(self, idx):
        chunk_idxs = [(self.frame_range[0] + idx * self.spacing) // self.chunk_size]
        return [(self.test_path if self.is_test else self.train_path) / f"{cidx}.npy" for cidx in chunk_idxs]

    def get_video_subsequence(self, video, idx):
        # Take a subsequence of the video.
        start_i = (self.frame_range[0] + idx * self.spacing) % self.chunk_size
        video = video[start_i:start_i+self.T]
        assert len(video) == self.T
        return video

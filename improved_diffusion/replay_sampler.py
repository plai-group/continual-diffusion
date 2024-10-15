import math
from typing import TypeVar, Optional, Iterator, Dict, Any
import blobfile as bf

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class DistributedReplaySampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        batch_size (int): Batch size associated with a single gradient step.
            Each GPU receives :attr:`batch_size//num_replicas` datapoints per gradient step.
        buffer_size (int, optional): Maximum replay buffer size. If none, there is no maximum.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedReplaySampler(dataset, 4) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, batch_size: int, buffer_size: Optional[int] = None,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None, seed: int = 0,
                 n_sequential: int = 1, save_args: Optional[Dict[str, Any]] = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be between [0, {num_replicas-1}]")
        self.num_replicas = num_replicas
        self.rank = rank
        self.save_args = save_args

        self.buffer_size = len(dataset) if buffer_size is None else buffer_size
        self.global_batch_size = batch_size  # 1 data stream, multiple buffer streams
        assert batch_size % num_replicas == 0
        self.local_batch_size = batch_size // num_replicas

        self.n_sequential = n_sequential
        self.start_index = 0
        self.next_index = self.start_index+1
        self.end_index = len(dataset)
        self.buffer_indices = []
        self.sample_generator = torch.Generator()
        self.update_generator = torch.Generator()
        self.sample_generator.manual_seed(seed+self.rank)
        self.update_generator.manual_seed(seed)

        self.buffer_indices = []

    def __iter__(self) -> Iterator[T_co]:
        """
        Some streams sequentially iterate the dataset while other streams perform reservoir sampling.
        """
        for i in range(self.start_index, self.end_index):

            # Depending on batch size, return
            for b in range(self.local_batch_size):
                # if (b == 0 and self.rank == 0) or i == 0:
                if i == 0 or self.rank * self.local_batch_size + b < self.n_sequential:
                    yield i  # main datastream
                else:
                    idx = torch.randint(len(self.buffer_indices), (1,), generator=self.sample_generator).item()
                    yield self.buffer_indices[idx]

            # update buffer
            if len(self.buffer_indices) < self.buffer_size:
                self.buffer_indices.append(i)
            elif self.buffer_size > 0:
                idx = torch.randint(i, (1,), generator=self.update_generator).item()
                if idx < len(self.buffer_indices):
                    self.buffer_indices[idx] = i

            self.next_index = i+1
            if self.save_args is not None and i % self.save_args["every"] == 0 and self.rank == 0:
                self.save_sampler(self.save_args["path"])

    def __len__(self) -> int:
        return self.end_index

    def load_sampler(self, path: str) -> None:
        """
        Given a save checkpoint, populate fields
        """
        loaded = torch.load(path)
        self.start_index = loaded["next_index"]
        self.next_index = self.start_index + 1
        self.buffer_indices = loaded["buffer_indices"]

    def save_sampler(self, path: str) -> None:
        """
        Given a save checkpoint, save fields
        """
        to_save = dict(next_index=self.next_index, buffer_indices=self.buffer_indices)
        with bf.BlobFile(path, "wb") as f:
            torch.save(to_save, f)

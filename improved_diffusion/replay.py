import torch
from typing import Any

from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler, TensorDictMaxValueWriter


class ReplayDataset:
    """
    A dataset class for replay buffers used in video diffusion modeling. Uses rolling update for context buffer and
    reservoir sampling for memory buffer.

    Args:
        context_size (int): The size of the context buffer.
        mem_size (int, optional): The size of the memory buffer. Defaults to 0.
        mem_batch_size (int, optional): The batch size for the memory buffer. Required if mem_size is not 0.

    Attributes:
        context_size (int): The size of the context buffer.
        mem_batch_size (int): The batch size for the memory buffer.
        context_buffer (ReplayBuffer): The buffer for storing context data.
        memory_buffer (ReplayBuffer): The buffer for storing memory data.
        has_mem (bool): Indicates whether the memory buffer is enabled.
        n_obs (int): The number of observations.

    Methods:
        update_context(data: Any) -> None: Updates the context buffer with new data.
        update_memory(data: Any) -> None: Updates the memory buffer with new data.
        update(data: Any) -> None: Updates both the context and memory buffers with new data.
        next_data() -> Tuple[context, memory]: Returns the next data from the context and memory buffers.
        get_context() -> torch.Tensor: Returns the context data from the context buffer.
        sample_memory() -> torch.Tensor: Returns a sample of memory data from the memory buffer.

    """

    def __init__(self, context_size, mem_size=0, mem_batch_size=None) -> None:
        self.context_size = context_size
        self.mem_batch_size = mem_batch_size
        self.context_buffer = ReplayBuffer(storage=LazyTensorStorage(context_size), batch_size=context_size,
                                           collate_fn=lambda x: x, sampler=SliceSampler(slice_len=context_size))
        if mem_size == 0:
            self.has_mem = False
            self.memory_buffer = None
        else:
            assert mem_batch_size is not None
            self.has_mem = True
            writer = TensorDictMaxValueWriter(rank_key="score")
            self.memory_buffer = ReplayBuffer(storage=LazyTensorStorage(mem_size), batch_size=mem_batch_size,
                                              collate_fn=lambda x: x, sampler=SliceSampler(slice_len=mem_batch_size),
                                              writer=writer)
        self.n_obs = 0

    def update_context(self, data: Any) -> None:
        """
        Updates the context buffer with new data.

        Args:
            data (Any): The new data to be added to the context buffer.

        Returns:
            None
        """
        batch_size, device =len(data), data.device
        data = TensorDict(dict(frames=data, time=self.n_obs * torch.ones(batch_size).to(device),
                               episode=torch.zeros(batch_size).to(device)), batch_size=len(data))
        self.context_buffer.extend(data)
        self.n_obs += 1

    def update_memory(self, data: Any) -> None:
        """
        Updates the memory buffer with new data.

        Args:
            data (Any): The new data to be added to the memory buffer.

        Returns:
            None
        """
        if self.has_mem:
            batch_size, device =len(data), data.device
            data = TensorDict(dict(frames=data,
                                   time=self.n_obs * torch.ones(batch_size).to(device),
                                   episode=torch.zeros(batch_size).to(device),
                                   score=torch.rand(batch_size).to(device),), batch_size=len(data))
            self.memory_buffer.extend(data)

    def update(self, data: Any) -> None:
        """
        Updates both the context and memory buffers with new data.

        Args:
            data (Any): The new data to be added to both buffers.

        Returns:
            None
        """
        self.update_context(data)
        self.update_memory(data)

    def get_context(self) -> torch.Tensor:
        """
        Returns the context data from the context buffer.

        Returns:
            torch.Tensor: The context data.
        """
        if len(self.context_buffer) <= self.context_size:
            # retrieve context then sort them by time index
            elements = self.context_buffer[:self.context_size]
            sorted_elements = sorted(elements, key=lambda x: x['time'])
            return torch.cat([el['frames'] for el in sorted_elements]).unsqueeze(0)
        else:  # never reached
            raise Exception("Context buffer size must not be greater than context size")

    def sample_memory(self) -> torch.Tensor:
        """
        Returns a sample of memory data from the memory buffer.

        Returns:
            torch.Tensor: The sample of memory data.
        """
        if not self.has_mem:
            return None
        elif len(self.memory_buffer) <= self.mem_batch_size:
            elements = self.memory_buffer[:self.mem_batch_size]
            # switch batch and time dimensions because we assume we get one frame per batch
            return elements['frames'].transpose(0, 1)
        else:
            elements = self.memory_buffer.sample()
            sorted_elements = sorted(elements, key=lambda x: x['time'])
            return torch.cat([el['frames'] for el in sorted_elements]).unsqueeze(0)

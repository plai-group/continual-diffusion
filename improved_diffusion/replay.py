from typing import Any

from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.envs.transforms import Compose, CatFrames


class ReplayDataset:
    def __init__(self, context_size, mem_size=0, mem_batch_size=None, time_dim=-4) -> None:
        self.context_buffer = ReplayBuffer(storage=LazyTensorStorage(context_size),
                                           transform=CatFrames(context_size, dim=time_dim),
                                           batch_size=context_size)
        if mem_size == 0:
            self.has_mem = False
            self.memory_buffer = None
        else:
            assert mem_batch_size is not None
            self.has_mem = True
            self.memory_buffer = ReplayBuffer(storage=LazyTensorStorage(mem_size),
                                              transform=CatFrames(mem_batch_size, dim=time_dim),
                                              batch_size=mem_batch_size)

    def update_context(self, data: Any) -> None:
        self.context_buffer.extend(data=data)

    def update_memory(self, data: Any) -> None:
        self.memory_buffer.extend(data=data)

    def update(self, data: Any) -> None:
        self.update_context(data)
        self.update_memory(data)

    def next_data(self):
        context = self.context_buffer.sample()
        memory = self.memory_buffer.sample() if self.has_mem else None
        return context, memory

    def get_context(self):
        return self.context_buffer.sample()

    def sample_memory(self):
        return self.memory_buffer.sample() if self.has_mem else None

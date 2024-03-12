import torch
from typing import Any

from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler


class ReplayDataset:
    def __init__(self, context_size, mem_size=0, mem_batch_size=None, time_dim=-4) -> None:
        self.context_size = context_size
        self.mem_batch_size = mem_batch_size
        self.context_buffer = ReplayBuffer(storage=LazyTensorStorage(context_size),
                                           #    transform=Compose(CatFrames(context_size, dim=time_dim, in_keys=["frame"])),
                                           # transform=CatFrames(context_size, dim=time_dim, in_keys=["frame"], out_keys=["frame"]),
                                           batch_size=context_size,  # context_size,
                                           collate_fn=lambda x: x,
                                           sampler=SliceSampler(slice_len=context_size))
        if mem_size == 0:
            self.has_mem = False
            self.memory_buffer = None
        else:
            assert mem_batch_size is not None
            self.has_mem = True
            self.memory_buffer = ReplayBuffer(storage=LazyTensorStorage(mem_size),
                                              # transform=CatFrames(mem_batch_size, dim=time_dim),
                                              batch_size=mem_batch_size,  # mem_batch_size,
                                              collate_fn=lambda x: x,
                                              sampler=SliceSampler(slice_len=mem_batch_size))
        self.n_obs = 0

    def update_context(self, data: Any) -> None:
        data = TensorDict(dict(frames=data, time=self.n_obs * torch.ones(len(data)).to(data.device),
                               episode=torch.zeros(len(data)).to(data.device)), batch_size=len(data))
        self.context_buffer.extend(data)
        self.n_obs += 1

    def update_memory(self, data: Any) -> None:
        data = TensorDict(dict(frames=data, time=self.n_obs * torch.ones(len(data)).to(data.device),
                               episode=torch.zeros(len(data)).to(data.device)), batch_size=len(data))
        self.memory_buffer.extend(data)

    def update(self, data: Any) -> None:
        self.update_context(data)
        self.update_memory(data)

    def next_data(self):
        context = self.context_buffer.sample()
        memory = self.memory_buffer.sample() if self.has_mem else None
        return context, memory

    def get_context(self):
        if len(self.context_buffer) < self.context_size:
            return self.context_buffer[:self.context_size]["frames"]
        else:
            return self.context_buffer.sample()["frames"]

    def sample_memory(self):
        # breakpoint()
        if not self.has_mem:
            return None
        elif len(self.memory_buffer) < self.mem_batch_size:
            return self.memory_buffer[:self.mem_batch_size]["frames"]
        else:
            return self.memory_buffer.sample()["frames"]

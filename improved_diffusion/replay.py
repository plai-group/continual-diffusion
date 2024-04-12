from tensordict import tensorclass
import torch
from typing import List



@tensorclass
class ReplayItem:
    frames: torch.Tensor
    absolute_index_map: torch.Tensor


class ReplayDataset:
    def __init__(self, stm_size, ltm_size=-1, mem_batch_size=None) -> None:
        self.stm_size = stm_size
        self.mem_batch_size = mem_batch_size
        self.mem_buffer_size = ltm_size // mem_batch_size if ltm_size > 0 else 0
        self._context_buffer: ReplayItem = None
        if ltm_size == -1:
            self.has_mem = False
            self._memory_buffer = None
        else:
            assert mem_batch_size is not None and ltm_size % mem_batch_size == 0 and ltm_size > mem_batch_size
            self.has_mem = True
            self._tmp_memory_buffer: List[ReplayItem] = []
            self._memory_buffer: List[ReplayItem] = []
        self.n_obs = 0

    def _update_context(self, item: ReplayItem) -> None:
        """
        Updates the context buffer with new data.

        Args:
            data (Any): The new data to be added to the context buffer.

        Returns:
            None
        """
        if self._context_buffer is None:
            self._context_buffer = item
        else:
            self._context_buffer = torch.cat((self._context_buffer, item), dim=1)

        if self._context_buffer.size(1) > self.stm_size:
            self._context_buffer = self._context_buffer[:, -self.stm_size:]

    def _update_memory(self, item: ReplayItem) -> None:
        """
        Updates the memory buffer with new data every self.mem_batch_size exemplars. Uses reservoir sampling.

        Args:
            data (Any): The new data to be added to the memory buffer.

        Returns:
            None
        """
        if not self.has_mem:
            return

        if len(self._tmp_memory_buffer) < self.mem_batch_size:
            self._tmp_memory_buffer.append(item)
            return

        item_group = torch.cat(self._tmp_memory_buffer, dim=1)
        if len(self._memory_buffer) < self.mem_buffer_size:
            self._memory_buffer.append(item_group)
        else:
            idx = torch.randint(self.n_obs // self.mem_batch_size, (1,))
            if idx < self.mem_buffer_size:
                self._memory_buffer[idx] = item_group

        self._tmp_memory_buffer = [item]

    def update(self, data: torch.Tensor) -> None:
        """
        Updates both the context and memory buffers with new data.

        Args:
            data (Any): The new data to be added to both buffers.

        Returns:
            None
        """
        item = self._data_to_replay_item(data)
        self.n_obs += data.size(1)  # update number of timesteps
        self._update_context(item)
        self._update_memory(item)

    def _data_to_replay_item(self, data: torch.Tensor) -> ReplayItem:
        B_new, T_new, device = data.size(0), data.size(1), data.device
        content = dict(frames=data, absolute_index_map=self.n_obs + torch.arange(T_new).repeat(B_new, 1).to(device))
        return ReplayItem(**content, batch_size=(B_new, T_new))

    def get_context(self) -> ReplayItem:
        """
        Returns the context data from the context buffer.

        Returns:
            torch.Tensor: The context data.
        """
        return self._context_buffer

    def sample_memory(self) -> ReplayItem:
        """
        Returns a sample of memory data from the memory buffer.

        Returns:
            torch.Tensor: The sample of memory data.
        """
        if not self.has_mem:
            return None
        elif len(self._memory_buffer) == 0:
            return self.get_context()
        else:
            return self._memory_buffer[torch.randint(len(self._memory_buffer), (1,))]

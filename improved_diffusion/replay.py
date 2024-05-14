from tensordict import tensorclass
import json
import os
import pathlib
import shutil
import torch
from typing import List, Union



@tensorclass
class ReplayItem:
    frames: torch.Tensor
    absolute_index_map: torch.Tensor


def maybe_load_item_from_disk(item: Union[ReplayItem, int], path_prefix: str):
    if isinstance(item, ReplayItem):
        return item
    else:
        return ReplayItem.load_memmap(os.path.join(path_prefix, str(item)))


class ReplayDataset:
    def __init__(self, stm_size, ltm_size=-1, mem_batch_size=None, n_sample_stm=1, n_sample_ltm=1, save_dir='/tmp') -> None:
        self.stm_size = stm_size
        self.mem_batch_size = mem_batch_size
        self.mem_buffer_size = ltm_size // mem_batch_size if ltm_size > 0 else 0
        self.n_sample_stm = n_sample_stm
        self.n_sample_ltm = n_sample_ltm

        self._stm_path = f"{save_dir}/context"
        self._tmp_ltm_path = f"{save_dir}/tmp_memory"
        self._ltm_path = f"{save_dir}/memory"
        self._json_path = f"{save_dir}/replay_state.json"

        self._context_buffer: ReplayItem = None
        if ltm_size == -1:
            self.has_mem = False
            self._memory_buffer = None
        else:
            assert mem_batch_size is not None and ltm_size % mem_batch_size == 0 and ltm_size > mem_batch_size
            self.has_mem = True
            self._tmp_memory_buffer: List[ReplayItem] = []
            self._memory_buffer: List[Union[ReplayItem, int]] = []
        self.n_obs = 0

    def load_state(self, load_dir: str):
        # Load replay information from last saved run state.
        self._stm_path = f"{load_dir}/context"
        self._tmp_ltm_path = f"{load_dir}/tmp_memory"
        self._ltm_path = f"{load_dir}/memory"
        self._json_path = f"{load_dir}/replay_state.json"

        with open(self._json_path) as f:
            config = json.load(f)
            self.n_obs = config['n_obs']

        mm_context_buffer = ReplayItem.load_memmap(self._stm_path)
        self._context_buffer = ReplayItem(**mm_context_buffer.to_tensordict(), batch_size=mm_context_buffer.batch_size)

        def load_disk_to_list(path: str, destination: List[Union[ReplayItem, int]], save_space: bool=False):
            dirnames = os.listdir(path)
            sorted_dirnames = sorted(dirnames, key=lambda e: int(e))
            for dirname in sorted_dirnames:
                load_path = os.path.join(path, dirname)
                entry: ReplayItem = ReplayItem.load_memmap(load_path)
                if save_space:
                    entry = entry.absolute_index_map[..., 0].item()
                destination.append(entry)

        if self.has_mem:
            load_disk_to_list(self._tmp_ltm_path, self._tmp_memory_buffer)
            load_disk_to_list(self._ltm_path, self._memory_buffer, save_space=True)
        print(f"loaded replay dataset state (after having observed {self.n_obs} datapoints) from disk.")

    def save_state(self):
        # Update replay information to the latest state.
        pathlib.Path(self._stm_path).mkdir(parents=True, exist_ok=True)
        self._context_buffer.memmap(self._stm_path, num_threads=0)

        def sync_disk_to_list(path: str, source: List[Union[ReplayItem, int]], save_space: bool = False):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            persisted_file_idxs = set()
            for i, entry in enumerate(source):
                if isinstance(entry, int):
                    entry = ReplayItem.load_memmap(os.path.join(path, str(entry)))
                file_idx = entry.absolute_index_map[..., 0].item()
                save_path = os.path.join(path, str(file_idx))
                entry.memmap(save_path, num_threads=0)
                if save_space:
                    source[i] = file_idx
                persisted_file_idxs.add(file_idx)
            sorted_file_idxs = [int(e) for e in sorted(os.listdir(path), key=lambda e: int(e))]
            for file_idx in sorted_file_idxs:
                load_path = os.path.join(path, str(file_idx))
                if file_idx not in persisted_file_idxs:
                    shutil.rmtree(load_path, ignore_errors=True)

        if self.has_mem:
            sync_disk_to_list(self._tmp_ltm_path, self._tmp_memory_buffer)
            sync_disk_to_list(self._ltm_path, self._memory_buffer, save_space=True)

        config = {'n_obs': self.n_obs}
        with open(self._json_path, "w") as f:
            json.dump(config, f, indent=2)

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

    def get_context(self, n_sample: int = None) -> ReplayItem:
        """
        Returns the context data from the context buffer.

        Returns:
            torch.Tensor: The context data.
        """
        n_sample = self.n_sample_stm if n_sample is None else n_sample
        return self._context_buffer.expand(n_sample, self._context_buffer.size(1))

    def sample_memory(self, n_sample: int = None) -> ReplayItem:
        """
        Returns a sample of memory data from the memory buffer.

        Returns:
            torch.Tensor: The sample of memory data.
        """
        n_sample = self.n_sample_ltm if n_sample is None else n_sample
        if not self.has_mem:
            return None
        elif len(self._memory_buffer) == 0:
            return self.get_context(n_sample=n_sample)
        else:
            items = [self._memory_buffer[i] for i in torch.randint(len(self._memory_buffer), (n_sample,))]
            items = [maybe_load_item_from_disk(item, self._ltm_path) for item in items]
            return torch.cat(items, dim=0)

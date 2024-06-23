import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_tokens(file_path):
    # Placeholder for token loading logic, assuming it returns a numpy array
    return np.load(file_path, mmap_mode='r')

class FineWebDataset(Dataset):
    def __init__(self, data_root, B, T, process_rank, num_processes, split='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        # Get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no shards found for split {split}"
        
        print(f"Found {len(self.shards)} shards for split {split}")
        
        self._reset()
    def _reset(self):
        # State, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    def __len__(self):
        # Length calculation
        return sum(len(load_tokens(shard)) for shard in self.shards) // (self.B * self.T) -1
    
    def __getitem__(self, idx):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = torch.tensor(buf[:-1].reshape(B, T), dtype=torch.long)  # Inputs
        y = torch.tensor(buf[1:].reshape(B, T), dtype=torch.long)   # Targets
        
        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes
        
        # If loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    def find_buffer_index(self, idx):
        cumulative_length = 0
        for i, buffer in enumerate(self.token_buffers):
            if idx < cumulative_length + len(buffer):
                return i, idx - cumulative_length
            cumulative_length += len(buffer)
        raise IndexError("Index out of bounds")
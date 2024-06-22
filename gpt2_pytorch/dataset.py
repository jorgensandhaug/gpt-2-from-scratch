import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FineWebDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if split in f]
        self.token_buffers = [np.load(f, mmap_mode='r') for f in self.files]

    def __len__(self):
        return sum(len(buffer) for buffer in self.token_buffers)

    def __getitem__(self, idx):
        # Find the correct file and the correct position within that file
        file_idx, token_idx = self._locate_index(idx)
        return torch.tensor(self.token_buffers[file_idx][token_idx], dtype=torch.long)

    def _locate_index(self, idx):
        # Locate the file and index within that file for the given index
        cumulative_length = 0
        for i, buffer in enumerate(self.token_buffers):
            if idx < cumulative_length + len(buffer):
                return i, idx - cumulative_length
            cumulative_length += len(buffer)
        raise IndexError("Index out of bounds")

# Usage example
data_dir = 'edu_fineweb10B'
train_dataset = FineWebDataset(data_dir, split='train')
val_dataset = FineWebDataset(data_dir, split='val')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

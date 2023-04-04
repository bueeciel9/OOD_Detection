import os
import glob
import pickle

import torch
from torch.utils.data import ConcatDataset, Dataset
import numpy as np


class CustomData(Dataset):
    def __init__(self, data_path,
                 normalize=True,
                 target_type=False,
                 augment=False,
                 device=torch.device('cpu')):
        data = torch.from_numpy(np.loadtxt(data_path, delimiter=',', skiprows=1))
        self.data_x = data[:, :-1].to(device=device, dtype=torch.float32)
        self.data_type = data[:, -1].to(device=device, dtype=torch.int32)

        if normalize:
            data_norm = torch.from_numpy(np.loadtxt('data/train_data_.csv', delimiter=',', skiprows=1))
            data_norm_x = data_norm[:, :-1].to(device=device, dtype=torch.float32)
            data_norm_x_min = data_norm_x.min(0).values
            data_norm_x_max = data_norm_x.max(0).values
            self.data_x = (self.data_x - data_norm_x_min) / (0.0001 + data_norm_x_max - data_norm_x_min)

        if augment:
            hp = torch.tensor([30, 20, 10, 50, 30, 30, 30, 30], dtype=torch.float32)
            if normalize:
                hp = (hp - hp.mean()) / hp.std()
            data_hp = hp[self.data_type.to(torch.long)]
            self.data_x = torch.cat([self.data_x, data_hp.reshape(-1, 1)], dim=-1)

        if isinstance(target_type, int):
            self.data_x = self.data_x[self.data_type == target_type]
            self.data_type = self.data_type[self.data_type == target_type]

        return

    def __getitem__(self, index):
        return self.data_x[index], self.data_type[index]

    def __len__(self):
        return len(self.data_x)


def collate_custom(batch):
    data_x, data_type = zip(*batch)
    return torch.stack(data_x), torch.stack(data_type)


def load_dataset(load_dir, batch_size, target_type=None, train_data=True, drop_last=False):
    assert os.path.exists(load_dir)
    print(f'Load data from: {load_dir}')
    final_dataset = CustomData(data_path=load_dir, target_type=target_type)
    print(f'Total number of data: {len(final_dataset)}')
    shuffle = True if train_data else False

    return torch.utils.data.DataLoader(final_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=drop_last,
                                       collate_fn=collate_custom)

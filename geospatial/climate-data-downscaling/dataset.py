import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import glob
import os
import cv2 

class ClimateDataset(Dataset):
    """
    PyTorch Dataset for paired low-res & high-res climate data.

    Args:
        lr_paths (list[str]): Paths of low-resolution .nc files
        hr_paths (list[str]): Paths of high-resolution .nc files
        mean (array): Per-channel mean (C,)
        std (array): Per-channel std (C,)
        size (int): Resize target (default=128)
        vars (list[str], optional): Variable names to keep (if None, use all)
    """
    def __init__(self, lr_paths, hr_paths, mean, std, size=128, vars=None):
        self.lr_paths = sorted(lr_paths)
        self.hr_paths = sorted(hr_paths)
        assert len(self.lr_paths) == len(self.hr_paths), "LR and HR files must match"
        
        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std  = np.array(std).reshape(-1, 1, 1)
        self.size = size
        self.vars = vars

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        # load LR 
        lr_ds = xr.open_dataset(self.lr_paths[idx])
        lr_vars = self.vars or list(lr_ds.data_vars)
        lr = [lr_ds[var].values.squeeze() for var in lr_vars]  # list of (H, W)
        lr = np.stack(lr, axis=0)[0]  # (C, H, W)
        lr_ds.close()

        # load HR
        hr_ds = xr.open_dataset(self.hr_paths[idx])
        hr = hr_ds[list(hr_ds.data_vars)[0]].values.squeeze()  # (H, W)
        hr_ds.close()
        hr = np.expand_dims(hr, axis=0)  # (1, H, W)

        # resize
        lr_resized = np.stack([cv2.resize(ch, (self.size, self.size), interpolation=cv2.INTER_CUBIC) for ch in lr], axis=0)

        # standardize
        lr_norm = (lr_resized - self.mean) / (self.std + 1e-6)

        # convert to torch
        lr_tensor = torch.tensor(lr_norm, dtype=torch.float32)     # (C, H, W)
        hr_tensor = torch.tensor(hr, dtype=torch.float32)  # (1, H, W)

        return lr_tensor, hr_tensor
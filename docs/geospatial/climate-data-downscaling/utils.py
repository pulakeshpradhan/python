import os
import glob
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import logging
import torch

def load_nc_file(path):
    """Load one NetCDF file and return stacked channels (C, H, W)."""
    ds = xr.open_dataset(path)
    arr = [ds[var].values.squeeze() for var in ds.data_vars]
    arr = np.stack(arr, axis=0)[0]  # (C, H, W)
    ds.close()
    return arr

def compute_mean_std(image_paths, num_samples=100, num_workers=4):
    """
    Compute per-channel mean/std from multiple NetCDF images.
    
    Args:
        image_paths (list[str]): Paths of .nc files
        num_samples (int): Number of random files to use
        num_workers (int): Parallel workers
    
    Returns:
        mean (C,), std (C,)
    """
    # Collect all .nc paths
    sample_image_paths = np.random.choice(image_paths, 
                                          size=min(num_samples, len(image_paths)), 
                                          replace=False)

    # Parallel load
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        arrays = list(executor.map(load_nc_file, sample_image_paths))

    # Stack all (N, C, H, W)
    stacked = np.stack(arrays, axis=0)

    # Compute per-channel stats
    mean = stacked.mean(axis=(0, 2, 3))  # (C,)
    std  = stacked.std(axis=(0, 2, 3))   # (C,)

    return mean, std


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, delta=0.0, path="checkpoint.pth"):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def setup_logger(log_file="train.log"):
    """Setup logger (both console + file)."""
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    # Clear previous handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

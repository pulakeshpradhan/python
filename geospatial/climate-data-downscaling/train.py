import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from tqdm import tqdm
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from utils import compute_mean_std, EarlyStopping, setup_logger
import config as cfg
from dataset import ClimateDataset
from model import QuantileDownscaler
from loss import QuantileLoss

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)





# Collect and sort file paths
lr_paths = sorted(glob(os.path.join(cfg.DATA_DIR, 'lr_images', '*.nc')))
hr_paths = sorted(glob(os.path.join(cfg.DATA_DIR, 'hr_images', '*.nc')))
assert len(lr_paths) == len(hr_paths), "LR and HR directories must contain same number of files"

# Random shuffle of indices
rng = np.random.default_rng(42)
indices = np.arange(len(lr_paths))
rng.shuffle(indices)

# 80:20 split index
split_idx = int(0.8 * len(indices))

train_idx = indices[:split_idx]
test_idx  = indices[split_idx:]

# Gather file lists
train_lr_paths = [lr_paths[i] for i in train_idx]
train_hr_paths = [hr_paths[i] for i in train_idx]

test_lr_paths = [lr_paths[i] for i in test_idx]
test_hr_paths = [hr_paths[i] for i in test_idx]

print(f"Train: {len(train_lr_paths)}, Test: {len(test_lr_paths)}")
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/deep-learning-from-scratch/[2] SRGAN/test.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: geo
    language: python
    name: geo
---

## Import libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import os
from glob import glob

import torch
from torch import nn
from model import Generator, Discriminator

DATA_DIR = '/beegfs/halder/DATA/climate_data_(kaushik)/'
```

## Model


<img src='https://imgs.search.brave.com/F5G4AmheMJCgwuV1QNcTqaWYLQ06u7_8iVcO76jG7bI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9naXRo/dWIuY29tL21zZWl0/emVyL3NyZ2FuL3Jh/dy9tYXN0ZXIvaW1h/Z2VzL2FyY2hpdGVj/dHVyZS5wbmc'>

```python
X = torch.randn((1, 17, 128, 128))

generator = Generator(in_channels=17, out_channels=1, num_channels=64, num_blocks=16)

X_hat = generator(X)
print(X_hat.shape)

discriminator = Discriminator(in_channels=1)
X_hat = discriminator(X_hat)
print(X_hat.shape)
```

```python
image_dir = os.path.join(DATA_DIR, 'images')
label_dir = os.path.join(DATA_DIR, 'labels')

image_paths = sorted(glob(os.path.join(image_dir, '*.nc')))
label_paths = sorted(glob(os.path.join(label_dir, '*.nc')))

image = xr.open_dataset(image_paths[0])
image = [image[var].values.squeeze() for var in image.data_vars]
image = np.stack(image, axis=0)

label = xr.open_dataset(label_paths[0])
# label = [label[var].values.squeeze() for var in label.data_vars]
# label = np.stack(label, axis=0)
```

```python
label
```

```python
image.shape
```

```python
plt.imshow(image[0, 12]);
```

```python
plt.imshow(label[0]);
```

```python
# Extract all data variables and drop the singleton time dimension
arrays = [image[var].values.squeeze() for var in ds.data_vars]

# Stack along channel axis
data = np.stack(arrays, axis=0)

print("Shape:", data.shape)  
# (channels, lat, lon)
```

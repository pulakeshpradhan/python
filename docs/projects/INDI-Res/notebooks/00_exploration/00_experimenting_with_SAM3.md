[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/INDI-Res/notebooks/00_exploration/00_experimenting_with_SAM3.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: torch
    language: python
    name: python3
---

## Import libraries

```python
# %pip install "segment-geospatial[samgeo3]"
```

```python
import os
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from samgeo import SamGeo3

root_dir = "/beegfs/halder/GITHUB/RESEARCH/INDI-Res/"
os.chdir(root_dir)
load_dotenv()
```

## Define the file path

```python
waterbody_file_path = os.path.join(root_dir, 'data', 'raw', 'waterbody.tif')
field_boundary_file_path = os.path.join(root_dir, 'data', 'raw', 'field_boundary.tif')

with rio.open(waterbody_file_path, 'r') as src:
    waterbody_image = src.read([1, 2, 3])
    waterbody_meta = src.meta
    
with rio.open(field_boundary_file_path, 'r') as src:
    field_boundary_image = src.read([1, 2, 3])
    field_boundary_meta = src.meta
```

```python
# Plot the images
# Convert to channel-last
waterbody_rgb = np.transpose(waterbody_image, (1, 2, 0))
field_boundary_rgb = np.transpose(field_boundary_image, (1, 2, 0))

def normalize_rgb(img):
    img = img.astype('float32')
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min)

waterbody_rgb = normalize_rgb(waterbody_rgb)
field_boundary_rgb = normalize_rgb(field_boundary_rgb)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(waterbody_rgb)
axes[0].set_title("Waterbody (RGB)")
axes[0].axis("off")

axes[1].imshow(field_boundary_rgb)
axes[1].set_title("Field Boundary (RGB)")
axes[1].axis("off")

plt.tight_layout()
plt.show()
```

## Request access to SAM3

```python
from huggingface_hub import login
login(token=os.getenv('HF_TOKEN'))
```

## Initiailize SAM3

```python
sam3 = SamGeo3(backend='transformers', device=None, checkpoint_path=None, load_from_HF=True)
```

## Set the image

```python
sam3.set_image(waterbody_file_path)
```

## Generate masks with text prompt

```python
sam3.generate_masks(prompt="waterbody")
```

```python
sam3.show_anns()
```

```python
sam3.show_masks()
```

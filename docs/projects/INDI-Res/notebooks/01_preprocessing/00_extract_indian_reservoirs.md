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
    name: python3
---

## Import libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os

root_dir = "/beegfs/halder/GITHUB/RESEARCH/INDI-Res/"
os.chdir(root_dir)
data_dir = os.path.join(root_dir, 'data')
```

## Read the datasets

```python
# Read the shapefile of India
india_boundary_path = os.path.join(data_dir, 'external', 'India_Boundary', 'India_Country_Boundary.shp')
india_boundary = gpd.read_file(india_boundary_path)
print(india_boundary.shape)
india_boundary.head()
```

```python
# Read the Global DAM Watch v1.0 database
gdw_reservoirs_data = gpd.read_file(os.path.join(data_dir, 'external', 'GDW_v1_0', 'GDW_reservoirs_v1_0.shp'))
gdw_reservoirs_data = gdw_reservoirs_data.to_crs(crs=india_boundary.crs)

gdw_barriers_data = gpd.read_file(os.path.join(data_dir, 'external', 'GDW_v1_0', 'GDW_barriers_v1_0.shp'))
gdw_barriers_data = gdw_barriers_data.to_crs(crs=india_boundary.crs)

# Subset the data for India
gdw_reservoirs_subset = gpd.sjoin(left_df=gdw_reservoirs_data, right_df=india_boundary, predicate='intersects')
gdw_barriers_subset = gpd.sjoin(left_df=gdw_barriers_data, right_df=india_boundary, predicate='intersects')

gdw_reservoirs_subset = gdw_reservoirs_subset.iloc[:, :-2].reset_index(drop=True)
gdw_barriers_subset = gdw_barriers_subset.iloc[:, :-2].reset_index(drop=True)
gdw_reservoirs_bounds = gdw_reservoirs_subset.copy()
gdw_reservoirs_bounds['geometry'] = gdw_reservoirs_bounds.geometry.envelope # bounds for each reservoir

print(gdw_reservoirs_subset.shape, gdw_reservoirs_bounds.shape, gdw_barriers_subset.shape)
gdw_reservoirs_subset.head()
```

## Plot the data

```python
# Plot the data
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
axes = axes.flatten()

india_boundary.plot(ax=axes[0], color='none', edgecolor='k', linewidth=1)
gdw_reservoirs_subset.plot(ax=axes[0], color='blue')
axes[0].set_title('GDW Reservoirs')

india_boundary.plot(ax=axes[1], color='none', edgecolor='k', linewidth=1)
gdw_barriers_subset.plot(ax=axes[1], color='red', markersize=0.5, alpha=0.5)
axes[1].set_title('GDW Barriers')

india_boundary.plot(ax=axes[2], color='none', edgecolor='k', linewidth=1)
gdw_reservoirs_bounds.plot(ax=axes[2], color='blue', markersize=0.5, alpha=0.5)
axes[2].set_title('GDW Reservoirs Bounds')

plt.tight_layout()
plt.show()
```

## Save the data

```python
# gdw_reservoirs_subset.to_file(os.path.join(data_dir, 'processed', 'GDW_reservoirs_v1_0_IN.gpkg'))
# gdw_barriers_subset.to_file(os.path.join(data_dir, 'processed', 'GDW_barriers_v1_0_IN.gpkg'))
```

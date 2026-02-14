[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/INDI-Res/notebooks/00_exploration/01_data_extraction_from_STAC.ipynb)

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
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rioxarray
from odc.stac import load

root_dir = "/beegfs/halder/GITHUB/RESEARCH/INDI-Res/"
os.chdir(root_dir)
data_dir = os.path.join(root_dir, "data")
```

## Read the datasets


```python
# Read the shapefile of India
india_boundary_path = os.path.join(
    data_dir, "external", "India_Boundary", "India_Country_Boundary.shp"
)
india_boundary = gpd.read_file(india_boundary_path)
print(india_boundary.shape)
india_boundary.head()
```

```python
# Read the Global DAM Watch v1.0 database for India
gdw_reservoirs_IN = gpd.read_file(
    os.path.join(data_dir, "processed", "GDW_reservoirs_v1_0_IN.gpkg")
)
gdw_reservoirs_IN.geometry = gdw_reservoirs_IN.geometry.buffer(distance=1000)
gdw_reservoirs_IN["area"] = gdw_reservoirs_IN.area / 1e6
gdw_reservoirs_IN.sort_values(by="area", ascending=False, inplace=True)
gdw_reservoirs_IN.to_crs(crs="EPSG:4326", inplace=True)
print(gdw_reservoirs_IN.shape)
gdw_reservoirs_IN.head()
```

## Search the STAC API


```python
API_URL = "https://hda.data.destination-earth.eu/stac/v2"
client = pystac_client.Client.open(API_URL)

bbox = list(gdw_reservoirs_IN.iloc[1].geometry.bounds)

search = client.search(
    collections=["EO.ECMWF.DAT.CAMS_GLOBAL_RADIATIVE_FORCING_AUX"],
    bbox=bbox,
    datetime="2001-01-01/2002-01-01",
)

items = [pc.sign(i) for i in search.get_all_items()]
print(f"Found {len(items)} scenes matching your criteria.")
```

```python
# Get all collections
collections = client.get_collections()

# Print basic info about each collection
for col in collections:
    print(f"ID: {col.id}")
    print(f"Title: {col.title}")
    print(f"Description: {col.description}\n")
```

## Read the assets


```python
# Display the available assets for the first item
assets_df = pd.DataFrame.from_dict(items[0].assets, orient="index").reset_index()
assets_df.columns = ["asset", "href"]
print(assets_df.shape)
assets_df.head()
```

## Load the data


```python
# Load Data into Memory (Lazy Loading)
data = load(
    items=items,
    band=["green", "nir08"],
    bbox=bbox,
    resolution=30,
    groupby="solar_day",
    chunks={"x": 2048, "y": 2048},
)

data
```

## Calculate NDWI


```python
ndwi = (data.green - data.nir08) / (data.green + data.nir08)

print("Streaming pixels and plotting... this may take 10-20 seconds.")
ndWi_snapshot = ndwi.isel(time=0).compute()

plt.figure(figsize=(10, 8))
ndWi_snapshot.plot.imshow(cmap="Blues_r", robust=False)
# plt.title(f"NDWI: {data.time[0].dt.date.item()}")
plt.axis("off")
plt.show()
```

```python
ndwi = (data.green - data.nir08) / (data.green + data.nir08)

print("Streaming pixels and plotting... this may take 10-20 seconds.")
ndWi_snapshot = ndwi.isel(time=9).compute()

plt.figure(figsize=(10, 8))
ndWi_snapshot.plot.imshow(cmap="Blues_r", robust=False)
# plt.title(f"NDWI: {data.time[0].dt.date.item()}")
plt.axis("off")
plt.show()
```

```python
ndWi_snapshot
```

```python
ndwi = (data.green - data.nir08) / (data.green + data.nir08)

print("Streaming pixels and plotting... this may take 10-20 seconds.")
ndWi_snapshot = ndwi.isel(time=23).compute()

plt.figure(figsize=(10, 8))
ndWi_snapshot.plot.imshow(cmap="Blues_r", robust=False)
# plt.title(f"NDWI: {data.time[0].dt.date.item()}")
plt.axis("off")
plt.show()
```

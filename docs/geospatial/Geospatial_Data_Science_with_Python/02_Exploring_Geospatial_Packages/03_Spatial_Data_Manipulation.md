[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Geospatial_Data_Science_with_Python/02_Exploring_Geospatial_Packages/03_Spatial_Data_Manipulation.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# **Spatial Data Manipulations**


## **Importing the Required Libraries**

```python
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

## **Setting Up the Working Directory**

```python
# Checking the Current Working Directory
os.getcwd()
```

```python
# Changing the Current Working Directory
file_path = r"D:\Coding\Git Repository\Geospatial_Data_Science_with_Python\Datasets\Shapafiles"
os.chdir(file_path)
```

```python
# Checking the Current Working Directory
os.getcwd()
```

## **Reading Spatial Data with GeoPandas**

```python
# Reading the natural earth 
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Geospatial_Data_Science_with_Python/02_Exploring_Geospatial_Packages/04_Geocoding.ipynb)

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

# **Geocoding**


## **Importing Required Libraries**

```python
import os
import pandas as pd
import geopandas as gpd
import geopy
from geopandas.tools import geocode
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

## **Setting Up the Current Working Directory**

```python
# Checking the current working directory
os.getcwd()
```

```python
# Changing the current working directory
file_path = r"D:\Coding\Git Repository\Geospatial_Data_Science_with_Python\Datasets\CSVs"
os.chdir(file_path)
```

```python
# Checking the current working directory
os.getcwd()
```

## **Reading the CSV Files using Pandas**

```python
# Reading the Kolkata_City_Attractions.csv file using Pandas
csv_name = "\\Kolkata_City_Attractions.csv"
cityAttraction = pd.read_csv(file_path + csv_name)
```

```python
# Checking the first 5 rows of the pandas dataframe
cityAttraction.head()
```

```python
# Extracting the 'Address' column from the dataframe as a pandas series
addresses = cityAttraction["Address"]
```

```python
# Printing the addresses
addresses
```

## **Setting Up the API of Mapquest**

```python
# Defining the API Key
api_key = creds.mapquest_api_key
# Defining the provider
provider = "openmapquest"
```

## **Applying the Geocoding**

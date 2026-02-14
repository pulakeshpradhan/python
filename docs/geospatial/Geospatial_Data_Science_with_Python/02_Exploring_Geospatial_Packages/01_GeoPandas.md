[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Geospatial_Data_Science_with_Python/02_Exploring_Geospatial_Packages/01_GeoPandas.ipynb)

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

# **GeoPandas**
**Author: Krishnagopal Halder**<br>

Geopandas is an open-source Python library that extends the capabilities of the popular data analysis library, pandas, by adding geospatial data processing and manipulation capabilities. It provides a convenient and efficient way to work with geospatial data, such as points, lines, and polygons, within the pandas DataFrame structure.

Geopandas leverages the functionalities of other powerful geospatial libraries, including Shapely, Fiona, and Pyproj, to handle geometric operations, file I/O, and coordinate transformations, respectively. By integrating these libraries, Geopandas simplifies the process of reading, manipulating, analyzing, and visualizing geospatial data.


## **01. Importing Required Libraries**

```python
import os
import warnings
warnings.filterwarnings("ignore")
import geopandas as gpd
import matplotlib.pyplot as plt
```

## **02. Setting Up the Working Directory**

```python
# Checking the current working directory
os.getcwd()
```

```python
# Change the current working directory
file_path =  r"D:\Coding\Git Repository\Geospatial_Data_Science_with_Python\Datasets\Shapafiles"
os.chdir(file_path)
```

```python
# Checking the current working directory
os.getcwd()
```

## **03. Reading and Writing Spatial Data with GeoPandas**

```python
# Reading data from local file
land = gpd.read_file(file_path+"\\ne_10m_land.shp")
```

```python
# Reading data from URL
url = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_land.geojson"
land_url = gpd.read_file(url)
```

**Dataset Description:**<br>
* **TIGER/Line Shapefiles**: TIGER/Line Shapefiles, commonly referred to as TIGER shapefiles or simply TIGER files, are a set of geospatial data files provided by the United States Census Bureau. TIGER stands for Topologically Integrated Geographic Encoding and Referencing. In TIGER shapefiles, the "US TIGER State data" refers to the boundaries and associated attributes of individual states within the United States. It represents the geographic extent of each state and provides information about their administrative divisions. The core TIGER/Line Files and Shapefiles do not include demographic data, but they do contain geographic entity codes (GEOIDs) that can be linked to the Census Bureau’s demographic data, available on [data.census.gov.](https://data.census.gov/)

* **Core Based Statistical Areas (CBSA):** Core Based Statistical Areas (CBSAs) provides the geographic boundaries and point information for more than 900 statistical regions defined by the U.S. Office of Management and Budget. A CBSA represents a highly populated core area and adjacent communities that have a high degree of economic and social integration with the core. CBSAs consist of counties and county equivalents, and are defined in two categories: (1) Metropolitan Statistical Areas, (2) Micropolitan Statistical Areas.

```python
# Reading data stored in a zip file
zip_path1 = file_path + "\\tl_2021_us_state.zip"
us_state = gpd.read_file(zip_path1)

zip_path2 = file_path + "\\tl_2021_us_cbsa.zip"
us_cbsa = gpd.read_file(zip_path2)
```

## **04. Filtering the Data**

```python
# Printing the first 5 records of the us_state geodataframe
us_state.head()
```

```python
# Filtering the California from us_state file
california = us_state[us_state["NAME"]=="California"]
```

* **'mask' Parameter:** In Geopandas, the mask parameter is commonly used in spatial operations to select or filter specific geometries based on a spatial relationship with another geometry or a set of geometries.

```python
# Creating a new geodataframe that includes cbsa areas of California
ca_cbsas = gpd.read_file(file_path + "\\tl_2021_us_cbsa.zip", mask=california)
ca_cbsas.head()
```

```python
# Using a bouning box to filter the data
bounding_box = (-128.82239, 42.15933, -123.82246, 38.7)

# Filtering the us_cbsa data based on bounding box
ca_cbsas_bbox = gpd.read_file(file_path + "\\tl_2021_us_cbsa.zip", bbox=bounding_box)
ca_cbsas_bbox.head()
```

## **05. Writing the Data**

```python
# Define the output path
output_path = r"D:\Coding\Git Repository\Geospatial_Data_Science_with_Python\Datasets\Shapafiles"

# Writing the ca_cbsas data as a shapefile
ca_cbsas.to_file(output_path+"\ca_cbsas.shp")

# Writing the ca_cbsas data as GeoJSON
ca_cbsas.to_file(output_path+"\ca_cbsas.geojson", driver="GeoJSON")
```

## **06. Spatial Data Visualization**

```python
# Plotting the ca_cbsas data
fig, ax = plt.subplots(figsize=(8, 6))
ca_cbsas.plot(ax=ax)
plt.xlabel("Longitude (X)")
plt.ylabel("Latitude (Y)")
plt.title("Core Based Statistical Areas (CBSA) of California")
plt.show()
```

```python
# Plotting simple choropleth map on ca_cbsas data
fig, ax = plt.subplots(figsize=(8, 6))
ca_cbsas.plot(ax=ax,
              cmap="Spectral",
              column="ALAND",
              edgecolor="black",
              linewidth=0.5,
              legend=True
             )
plt.title("Choropleth Map showing Land Areas of California CBSAs\n")
plt.xlabel("Longitude (X)")
plt.ylabel("Latitude (Y)")
plt.show()
```

```python
# Plotting the choropleth map of world's population
world_pop = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world_pop.head()
```

```python
fig, ax = plt.subplots(figsize=(12, 10))
world_pop.plot(ax=ax,
               column="pop_est",
               cmap="Spectral",
               edgecolor="black",
               linewidth=0.5
              )
plt.title("Choropleth Map showing World Population Estimates")
plt.xlabel("Longitude (X)")
plt.ylabel("Latitude (Y)")
plt.show()
```

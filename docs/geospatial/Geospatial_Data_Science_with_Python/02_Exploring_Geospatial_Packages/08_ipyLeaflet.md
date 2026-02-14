[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Geospatial_Data_Science_with_Python/02_Exploring_Geospatial_Packages/08_ipyLeaflet.ipynb)

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

# **ipyLeaflet**
ipyleaflet is a Python library that provides interactive mapping capabilities in Jupyter notebooks and JupyterLab. It is built on top of the popular JavaScript mapping library, Leaflet.js. ipyleaflet allows you to create interactive maps, markers, polygons, layers, and other geospatial visualizations directly within the Jupyter environment.

With ipyleaflet, you can display maps with various base layers such as OpenStreetMap, Mapbox, and other tile layers. You can add markers and polygons to the map, customize their appearance, and interact with them. The library also supports layers like GeoJSON, WMS (Web Map Service), and Tile layers, allowing you to overlay additional data on the map.

One of the key features of ipyleaflet is its interactivity. You can zoom in and out, pan the map, and interact with markers and polygons using mouse events. ipyleaflet also provides widgets that allow you to control and manipulate the map dynamically, such as sliders, checkboxes, and dropdown menus.


## **01. Importing Required Libraries**

```python
import os
import ipyleaflet
import pandas as pd
import geopandas as gpd
from geopandas.tools import geocode
import geopy
```

## **02. Setting Up the Current Working Directory**

```python
# Checking the current working directory
os.getcwd()
```

```python
# Changing the current working directory
file_path = r"D:\Coding\Git Repository\Geospatial_Data_Science_with_Python\Datasets\CSVs"
os.chdir(file_path)
# Checking the current working directory
os.getcwd()
```

## **03. Reading the CSV File with Pandas**

```python
# Reading the Kolkata_City_Attractions.csv using pandas
city_attractions = pd.read_csv("Kolkata_City_Attractions.csv")
# Checking the rows of the csv
city_attractions.head(10)
```

## **04. Geocoding Addresses using Nominatim**

```python
# Geocoding the addresses using nominatim
kolkata_attractions_gpd = geocode(city_attractions["Address"], provider="nominatim", user_agent="nominatim")
# Check the geocoded addresses
kolkata_attractions_gpd
```

```python
# Joining the two tables
city_attractions_gpd = kolkata_attractions_gpd.join(city_attractions["Attraction"])
# Checking the geodataframe
city_attractions_gpd
```

## **05. Data Cleaning**

<!-- #region -->
**loc:**<br>
The loc method in Pandas is used to access and manipulate data based on label-based indexing. It allows you to select specific rows and columns from a DataFrame by specifying the labels of the rows and columns you want to retrieve.

The basic syntax for using the loc method is as follows:
```python
dataframe.loc[row_labels, column_labels]
```
* **row_labels:** This can be a single label, a list of labels, or a slice object specifying the rows you want to select. The labels can be either the index labels or boolean conditions applied to the index.
* **column_labels:** This can be a single label, a list of labels, or a slice object specifying the columns you want to select. The labels can be either the column names or boolean conditions applied to the columns.
<!-- #endregion -->

```python
# Rearranging the columns of the geodataframe
city_attractions_gpd = city_attractions_gpd.loc[:, ["Attraction", "address", "geometry"]]
# Checking the geodataframe
city_attractions_gpd
```

```python
# Removing the rows with null geometry
city_attractions_gpd.dropna(inplace=True)
# Resetting the index
city_attractions_gpd.reset_index(inplace=True, drop=True)
# Checking the geodataframe
city_attractions_gpd
```

```python
# Adding in lat and lon columns
city_attractions_gpd["lon"] = city_attractions_gpd["geometry"].x
city_attractions_gpd["lat"] = city_attractions_gpd["geometry"].y
```

```python
# Checking the geodataframe
city_attractions_gpd
```

## **06. Create an Interactive Map with ipyLeaflet**

```python
from ipyleaflet import Map, basemaps
```

```python
# Creating a map using ipyleaflet
city_map = Map(basemap = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik),
    		   center=(22.5726, 88.3639), 
    		   zoom=12
               )
# Displaying the map
city_map
```

**iterrows():**<br>
In geopandas, the iterrows() function allows you to iterate over the rows of a GeoDataFrame. It returns an iterator that yields both the index and the row data for each row in the GeoDataFrame.

```python
# Mapping the attractions
from ipyleaflet import Marker
for index, row in city_attractions_gpd.iterrows():
    marker = Marker(location=[row.loc["lat"], row.loc["lon"]], title=row.loc["Attraction"])
    city_map.add_layer(marker)

# Displaying the city map after adding marker points
city_map
```

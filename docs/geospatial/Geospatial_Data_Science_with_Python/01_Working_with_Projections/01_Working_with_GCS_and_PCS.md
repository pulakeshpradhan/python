[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Geospatial_Data_Science_with_Python/01_Working_with_Projections/01_Working_with_GCS_and_PCS.ipynb)

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

# **Working with Geographic and Projected Coordinate System**
**Author: Krishnagopal Halder**
1. **Geographic Coordinate System:** The geographic coordinate system (GCS) uses a three-dimensional spherical surface to represent the Earth's shape. It is based on a datum, which defines the position and orientation of the coordinate system with respect to the Earth. The GCS uses angular units (degrees) to express coordinates. The most common GCS is the WGS84 (World Geodetic System 1984), which is widely used for global positioning and mapping purposes.

2. **Projected Coordinate System:** A projected coordinate system (PCS) is a two-dimensional Cartesian coordinate system that flattens the Earth's surface onto a flat map. It uses a map projection to transform the spherical coordinates into x-y coordinates. Map projections mathematically convert the curved Earth's surface onto a flat surface, introducing distortions in distance, shape, area, or direction.


<center><img src="https://developers.arcgis.com/geoanalytics/static/a13dabcadbf901b072baf212523a6658/29492/gcs-pcs-location.png"> </center>


This notebook provides an overview of how to work with geographic and projected coordinate systems using Python. Understanding coordinate systems is crucial for geospatial data analysis, mapping, and spatial analysis tasks. This guide introduces the concepts of geographic and projected coordinate systems and demonstrates how to perform coordinate system transformations, conversions, and visualizations using Python libraries.


## **01. Importing Required Libraries**

```python
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

## **02. Setting Up the Working Directory**

```python
# Checking the current working directory
print(os.getcwd())
```

```python
# Changing the current working directory
file_path = r"D:\Coding\Git Repository\Geospatial_Data_Science_with_Python\Datasets\Shapafiles"
os.chdir(file_path)
# Checking the current working directory
print(os.getcwd())
```

```python
# Checking the files in the current working directory
print(os.listdir())
```

## **03. Reading Shapefiles with GeoPandas**
GeoPandas is an open-source Python library that extends the capabilities of the popular data manipulation library, pandas, to handle geospatial data. It provides an easy and intuitive way to work with geospatial data, combining the power of pandas' data manipulation and analysis with the geometric operations and spatial functionality of other geospatial libraries, such as Shapely and Fiona.

```python
# Reading a Natural Earth world shapefile with GeoPandas
world = gpd.read_file("ne_10m_land.shp")
# Reading a Natural Earth Populated Places Point shapefile
pop_cities = gpd.read_file("ne_10m_populated_places_simple.shp")
```

```python
# Checking the first five rows of the populated places dataset
pop_cities.head()
```

```python
# Extracting the Admin-0 capitals only
capitals = pop_cities[pop_cities["featurecla"]=="Admin-0 capital"]
```

```python
# Checking the Admin-0 capitals
capitals.head()
```

```python
# Reading a Natural Earth 10m interval Graticules shapefile
grat = gpd.read_file("ne_110m_graticules_10.shp")
```

## **04. Checking the Metadata of the CRS**

```python
# Checking the CRS metadata of the world shapefile
world.crs
```

```python
# Checking the CRS metadata of the cities shapefile
capitals.crs
```

```python
# Checking the CRS metadata of the graticules shapefile
grat.crs
```

```python
# Checking if all the shapefiles are in same CRS or not
world.crs == capitals.crs == grat.crs
```

## **05. Plotting the Shapefiles in a Map**

```python
fig, ax = plt.subplots(figsize=(12, 10))
world.plot(ax=ax, color="darkgray")
capitals.plot(ax=ax, color="black", markersize=8, label="Populated Places")
grat.plot(ax=ax, color="lightgray", linewidth=0.5)
ax.set(xlabel="Longitude(Degrees)", ylabel="Latitude(Degrees)", title="Populated Places showing in WGS 1984 Datum")
plt.legend()
plt.show()
```

## **06. Reprojecting the Data**
Reprojection, also known as coordinate transformation or coordinate conversion, refers to the process of converting spatial data from one coordinate system to another. It involves transforming the coordinates of geographic features from their original reference system to a different reference system, often with a different datum or projection.

Reprojection is necessary when working with geospatial data that is collected, stored, or analyzed in different coordinate systems. Each coordinate system has its own set of parameters, such as the datum, map projection, units of measurement, and spatial reference. Reprojection ensures that different datasets or layers align properly and can be integrated or overlaid accurately in a consistent spatial reference system.

```python
# Reprojecting the data from WGS 1984 CRS to Azimuthal Equidistant Projection
world_ae = world.to_crs("ESRI:54032")
capitals_ae = capitals.to_crs("ESRI:54032")
grat_ae = grat.to_crs("ESRI:54032")
```

```python
# Checking the reprojected CRS
world_ae.crs
```

```python
capitals_ae.crs
```

```python
grat_ae.crs
```

## **07. Plotting the Reprojected Shapefiles in a Map**

```python
# Creating a function that will take the files and plot them in a map
def plot_map_layers(gdf_1, gdf_2, gdf_3, title, unit, legend):
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_1.plot(ax=ax, color="darkgray")
    gdf_2.plot(ax=ax, color="black", markersize=8, label=legend)
    gdf_3.plot(ax=ax, color="lightgray", linewidth=0.5)
    ax.set(xlabel="X Coordinate-" + unit,
           ylabel="Y Coordinate-" + unit,
           title=title
           )
    plt.legend()
    plt.show()
```

```python
# Using the function to plot the map
map_title = "Populated Places showing in Azimuthal Equidistant Projection"
plot_map_layers(world_ae, capitals_ae, grat_ae, map_title, "Meters", "Poulated Places")
```

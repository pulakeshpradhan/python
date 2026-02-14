[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Geospatial_Data_Science_with_Python/02_Exploring_Geospatial_Packages/02_Spatial_Data_Structures.ipynb)

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

# **Spatial Data Structures and its Methods and Attributes**
In Geopandas, the spatial data structure is based on the GeoDataFrame, which is an extension of the pandas DataFrame with additional functionality to handle geospatial data. The spatial data structure in Geopandas allows you to work with spatially referenced data, such as points, lines, and polygons, within a tabular framework. A GeoDataFrame and a GeoSeries are two fundamental components of the Geopandas library, which extends the capabilities of pandas for working with geospatial data.

**1. GeoDataFrame:**
A GeoDataFrame is a tabular data structure that extends the functionality of pandas DataFrame by incorporating a "geometry" column. This column stores geometric objects associated with each row of the DataFrame. The geometries can represent points, lines, polygons, or other spatial entities.

**Key features:**

* **DataFrame Structure:** A GeoDataFrame retains the tabular structure of a pandas DataFrame, allowing for efficient indexing, filtering, and manipulation of both the attribute data and geometric information.
* **Geometry Column:** The "geometry" column in a GeoDataFrame contains the geometric objects representing the spatial features. It can store single geometries or collections of geometries.
* **Attribute Data:** A GeoDataFrame can have additional columns that store attribute data related to the spatial features. These columns can contain various information such as names, IDs, population figures, or any other relevant attributes.
* **Coordinate Reference System (CRS):** A GeoDataFrame includes information about the Coordinate Reference System, defining the spatial reference and coordinate system used by the geometries.
* **Integration with Spatial Operations:** The GeoDataFrame integrates with Geopandas' spatial operations, allowing for geometric manipulations, spatial joins, spatial queries, and other spatial analysis tasks.

**2. GeoSeries:**
A GeoSeries, on the other hand, is a one-dimensional array-like object that represents a series of geometric objects. It is based on pandas' Series but is specifically designed to handle spatial data.

**Key features:**

* **Series-like Behavior:** A GeoSeries shares many similarities with a pandas Series, such as indexing, slicing, and applying functions or operations element-wise.
* **Geometry Storage:** The primary purpose of a GeoSeries is to store and manage geometric objects. Each element of the series represents a single geometry, such as a point, line, or polygon.
* **Coordinate Reference System (CRS):** A GeoSeries also includes information about the Coordinate Reference System, providing spatial reference and coordinate system details for the geometries within the series.
* **Integration with Spatial Operations:** Similar to a GeoDataFrame, a GeoSeries integrates with Geopandas' spatial operations, allowing for geometric manipulations, spatial queries, and other spatial analysis tasks at the individual geometry level.


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
# Printing the current working directory
os.getcwd()
```

```python
# Changing the current working directory
file_path = r"D:\Coding\Git Repository\Geospatial_Data_Science_with_Python\Datasets\Shapefiles"
os.chdir(file_path)
```

```python
# Checking the current working directory again
os.getcwd()
```

```python
# Printing all the shapefile names of the current working directory
for item in os.listdir():
    if item.endswith(".shp"):
        print(item)
```

## **03. Reading Spatial Data with GeoPandas**

```python
# Reading the ne_10m_land dataset
land = gpd.read_file(file_path + "\\ne_10m_land.shp")
```

```python
# Checking the first 5 rows of the shapefile
land.head()
```

```python
# Checking the datatype of the land variable
type(land)
```

```python
# Checking the datatype of the geometry column of the land geodataframe
type(land["geometry"])
```

## **04. Attributes of GeoSeries Data Structure**

```python
# Defining the geometry column in a seperate variable
land_geometry = land["geometry"]
```

```python
# Checking the type of land_geometry_variable
type(land_geometry)
```

**crs:** <br>
The crs attribute stores the spatial reference and coordinate system information associated with the geometries in the GeoSeries. It provides metadata about how the coordinates are interpreted and projected in the real world.

```python
# Checking the CRS information of the geometry using crs() attribute
land_geometry.crs
```

**geom_type:**<br>
The geom_type attribute returns the geometry type of each geometry in the GeoSeries. It indicates whether each geometry is a point, line, polygon, or another geometric type.

```python
# Checking the geometry type of each feature in the GeoSeries
print(land_geometry.geom_type)
```

**area:**<br>
The area attribute calculates the area of each geometry in the GeoSeries. The area is computed based on the spatial reference system (CRS) of the GeoSeries.

```python
# Calculating the area of each geometry in the GeoSeries
land_geometry.area
```

**bounds:**<br>
The bounds attribute returns a bounding box (or minimum bounding rectangle) for each geometry in the GeoSeries. The bounding box represents the minimum and maximum x and y coordinates that enclose the geometry.

```python
# Checking the bounding box for each geometry
land_geometry.bounds
```

**total_bounds:**<br>
The total_bounds attribute returns the overall bounding box that encompasses all geometries in the GeoSeries. It provides the minimum and maximum x and y coordinates that cover the entire collection of geometries.

```python
# Checking the overall bounding box of the geometries
land_geometry.total_bounds
```

## **05. Methods of GeoSeries Data Structure**


**to_crs:**<br>
The to_crs method allows you to transform the coordinate reference system (CRS) of the geometries in a GeoSeries. It takes a CRS object or a string representation of a CRS as its argument. By applying to_crs, you can reproject the geometries to a different CRS, enabling spatial analysis and visualization in a consistent coordinate system.

```python
# Plotting the land_geometry map with the default WGS_84 Datum
fig, ax = plt.subplots(figsize=(8, 6))
land_geometry.plot(ax=ax)
plt.title("World Map showing in WGS84 Datum")
plt.xlabel("Longitude (X)")
plt.ylabel("Latitude (Y)")
plt.show()
```

```python
# Changing the CRS of the land_geometry to the Robinson Projection
reprojected_land = land_geometry.to_crs("ESRI:54030")
# Checking the reprojected CRS
reprojected_land.crs
```

```python
# Reading a 10 degree graticules file with geopandas
grat = gpd.read_file(file_path + "\\ne_110m_graticules_10.shp").to_crs("ESRI:54030")
# Plotting the reprojected_land
fig, ax = plt.subplots(figsize=(8, 6))
reprojected_land.plot(ax=ax, color="darkgray")
grat.plot(ax=ax, color="black", linewidth=0.2)
plt.title("World Map showing in Robinson Projection")
plt.xlabel("X Coordinate-Meters")
plt.ylabel("Y Coordinate-Meters")
plt.show()
```

**centroid:** <br>
The centroid method computes the centroid (geometric center) of each polygon geometry in a GeoSeries. It returns a new GeoSeries with the centroid points as geometries. This method is applicable only to GeoSeries containing polygon geometries.

```python
# Defining the bounding box for the North America
bounds = (-125.0, 24.0, -66.0, 49.0)
```

```python
# Reading the us_state shapefile with geopandas and filtering with the bounding box
us_state = gpd.read_file(file_path + "\\tl_2021_us_state.zip", bbox=bounds)
```

```python
# Checking the us_state GeoDataframe
us_state.head()
```

```python
# Checking the CRS of the us_state file
us_state.crs
```

```python
# Calculating the centroids of all the ploygons in the us_state data
us_centroids = us_state["geometry"].centroid
```

```python
# Plotting the us_state and us_centroids to a map
fig, ax = plt.subplots(figsize=(8, 6))
us_state.plot(ax=ax, 
              cmap="Set3", 
              column="GEOID",
              linewidth=0.3,
              edgecolor="black")
us_centroids.plot(ax=ax, 
                  color="red",
                  edgecolor="black",
                  markersize=10,
                  linewidth=0.5,
                  label="centroid")
plt.title("USA Map with State Centroids")
plt.xlabel("Longitude (X)")
plt.ylabel("Latitude (Y)")
plt.legend()
plt.show()
```

**distance:** <br>
The distance method calculates the Euclidean distance between each geometry in a GeoSeries and a provided geometry. The provided geometry can be a single point, line, or polygon. The method returns a new GeoSeries with the distances calculated for each geometry.

```python
# Adding the centroid values in a separate column in us_states GeoDataframe
us_state["centroid"] = us_centroids
```

```python
# Checking the column names after adding the centroid column
us_state.columns
```

```python
# Extracting the centroid value of the California state
california_centroid = us_state["centroid"][us_state["NAME"]=="California"].to_crs("EPSG:32610").values
# Extracting the centroid value of the Oregon state
oregon_centroid = us_state["centroid"][us_state["NAME"]=="Oregon"].to_crs("EPSG:32610").values
```

```python
# Checking the CRS of the california_centroid
california_centroid.crs
```

```python
# Printing the centroid value of the California and New York States
print("California Centroid:", california_centroid)
print("Oregon Centroid:", oregon_centroid)
```

```python
# Calculating the distance between California and New York using distance() method
CA_OR_distance = california_centroid.distance(oregon_centroid)
# Converting the distance into kilometers
distance_km = CA_OR_distance[0] / 1000
# Printing the distance
print("The euclidean distance between California and Oregon is", 
      round(distance_km, 2), 
      "Kilometers.")
```

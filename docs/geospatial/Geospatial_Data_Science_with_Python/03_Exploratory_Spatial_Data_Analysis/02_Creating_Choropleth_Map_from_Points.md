[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Geospatial_Data_Science_with_Python/03_Exploratory_Spatial_Data_Analysis/02_Creating_Choropleth_Map_from_Points.ipynb)

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

# **Creating Choropleth Map from Points**


## **01. Importing Required Libraries**

```python
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

## **02. Setting Up the Current Working Directory**

```python
# Checking the current working directory
os.getcwd()
```

```python
# Changing the current working directory
file_path = r"D:\Coding\Git Repository\Geospatial_Data_Science_with_Python\Datasets"
os.chdir(file_path)
# Checking the new current working directory
os.getcwd()
```

```python
# Defining the CSV and Shapefile paths
csv_path = file_path + "\\CSVs"
shp_path = file_path + "\\Shapefiles"
```

## **03. Reading the Data**

```python
# Reading the us_county data using geopandas
us_county = gpd.read_file(shp_path + "\\tl_2022_us_county.zip")
us_county.head()
```

```python
# Reading the California housing data
housing = pd.read_csv(csv_path + "\\housing.csv")
housing.head()
```

## **04. Prepairing the Dataset**


```python
# Checking the CRS of the us_county data
us_county.crs
```

```python
# Changing the CRS from NAD83 to WGS84
us_county = us_county.to_crs(4326)
# Checking the new CRS of the data
us_county.crs
```

```python
# Subsetting the US Counties that belong to California state
# STATFP of California = 06
ca_county = us_county.loc[:,:][us_county["STATEFP"]=="06"]
# Checking the ca_county dataframe
ca_county.head()
```

```python
# Printing the housing.csv data
housing.head()
```

```python
# Converting the housing dataframe into geodataframe
housing_gdf = gpd.GeoDataFrame(housing, 
                               crs=4326, 
                               geometry=gpd.points_from_xy(housing.longitude, housing.latitude))
# Checking the geodataframe
housing_gdf.head()
```

<!-- #region -->
In Geopandas, spatial join and aggregation are powerful operations used to combine geospatial datasets based on their spatial relationships and perform calculations or summaries on the resulting joined data. Let's explore each of these operations:

**Spatial Join:**<br>
Spatial join is the process of combining two or more geospatial datasets based on their spatial relationships. It involves linking the attributes of one dataset to another dataset based on the spatial proximity or overlap of their geometries. The resulting joined dataset contains attributes from both datasets.

In Geopandas, the sjoin() function is used to perform a spatial join. It takes two GeoDataFrames as input and joins them based on a specified spatial relationship, such as "intersects," "contains," or "within." The function returns a new GeoDataFrame with the attributes from both input datasets.

* **Parameters:**<br>
1. **how:** This parameter determines how the spatial join is performed. The available options are:
   
   * "inner" (default): Only the intersecting or overlapping geometries between the two datasets are retained in the joined result.
   * "left": All features from the left GeoDataFrame are kept, and attributes from the right GeoDataFrame are added to the resulting joined dataset based on the spatial relationship.
   * "right": All features from the right GeoDataFrame are kept, and attributes from the left GeoDataFrame are added to the resulting joined dataset based on the spatial relationship.
   * "outer": All features from both GeoDataFrames are retained, and attributes are added based on the spatial relationship. Non-overlapping features will have missing attribute values.<br>


2. **op:** This parameter specifies the spatial relationship used for the join. The available options are:

   * "intersects" (default): Joins geometries if they intersect or have any spatial overlap.
   * "contains": Joins geometries if one completely contains the other.
   * "within": Joins geometries if one is completely within the other.
   * "touches": Joins geometries if they share a common boundary but do not overlap.
<!-- #endregion -->

```python
# Applying Spatial Join between ca_county and housing_gdf
county_housing_sj = gpd.sjoin(ca_county, housing_gdf, how="left", op="contains")
# Checking the spatially joined dataframe
county_housing_sj.head()
```

```python
# Selecting required columns only
county_housing_sj = county_housing_sj[["GEOID", "NAME", "median_house_value", "geometry"]]
```

**Aggregation:**<br>
Aggregation in Geopandas involves summarizing or calculating statistics on a group of features based on a specific spatial unit. It allows you to aggregate or dissolve geometries together and compute summary statistics for the combined features.

The dissolve() function in Geopandas is commonly used for aggregation. It combines geometries that share a common attribute value and calculates summary statistics for the grouped features. You can specify which attribute to dissolve by, and the function can compute various statistics, such as the sum, mean, maximum, minimum, or count of a specific attribute

```python
# Aggregating the rows by GEOID
county_housing_agg = county_housing_sj.dissolve(by=["GEOID"], aggfunc="mean")
```

```python
# Checking the aggregated geodataframe
county_housing_agg.head()
```

```python
# Manipulating the aggregated data
county_housing_agg.reset_index(inplace=True)
county_housing_agg = county_housing_agg.loc[:, ["GEOID", "median_house_value", "geometry"]]
# Checking the manipulated aggregated dataframe
county_housing_agg.head()
```

## **05. Spatial Data Visualization**

```python
# Plotting a choropleth map using geopandas
axes = county_housing_agg.plot(column="median_house_value",
                               cmap="Reds",
                               edgecolor="black",
                               linewidth=0.5,
                               legend=True)
axes.set_xlabel("Longitude")
axes.set_ylabel("Latitude")
axes.set_title("Choropleth Map of California Counties' Median House Value in US$")
axes.title.set_size(10)
```

**Geoviews:**
Geoviews is a Python library that is built on top of the powerful visualization library called HoloViews. It provides a high-level interface for creating interactive and declarative visualizations of geospatial data. Geoviews is designed to simplify the process of working with complex geospatial datasets and enables easy exploration and analysis of geographic information.

Key features and capabilities of Geoviews include:

* **Declarative syntax:** Geoviews allows you to define visualizations using a concise and declarative syntax, which makes it easy to create complex plots with minimal code. You can specify the data, visual attributes, and other parameters using a chainable syntax.

* **Seamless integration:** Geoviews seamlessly integrates with popular geospatial libraries such as GeoPandas, Cartopy, and Bokeh. It can ingest data from different formats and sources, including shapefiles, GeoJSON, and raster datasets.

* **Interactive visualizations:** Geoviews supports interactive exploration and visualization of geospatial data. It leverages the interactivity features of HoloViews and Bokeh, allowing you to zoom, pan, and explore the data with tooltips, hover effects, and interactive widgets.

* **Dynamic overlays:** Geoviews enables the creation of dynamic overlays, where multiple layers of geospatial data can be combined and visualized together. This makes it easy to overlay different data types, such as points, lines, polygons, and raster images.

* **Geospatial projections:** Geoviews supports various map projections, allowing you to display and transform geographic data in different coordinate reference systems (CRS). It provides a consistent interface for working with different projections, making it easy to switch between them.

* **Geospatial operations:** Geoviews provides functionality for performing common geospatial operations, such as spatial aggregation, spatial joins, and spatial filtering. These operations can be used to analyze and process geospatial data effectively.

```python
# Importing geoviews library
import geoviews
```

```python
# Plotting an interactive choropleth map using geoviews
geoviews.extension("bokeh")
choropleth = geoviews.Polygons(data=county_housing_agg, vdims=["median_house_value", "GEOID"])
choropleth.options(height=600,
                   width=500,
                   title="Choropleth Map of California Counties' Median House Value in US$",
                   tools=["hover"],
                   cmap="Reds",
                   colorbar=True,
                   colorbar_position = "bottom")
```

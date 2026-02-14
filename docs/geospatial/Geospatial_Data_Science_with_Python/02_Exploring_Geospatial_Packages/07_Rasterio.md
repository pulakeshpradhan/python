[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Geospatial_Data_Science_with_Python/02_Exploring_Geospatial_Packages/07_Rasterio.ipynb)

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

# **Rasterio**
Rasterio is a Python library designed to handle geospatial raster datasets. It provides a powerful and efficient way to read, write, manipulate, and analyze raster data. Rasterio builds upon the capabilities of the GDAL (Geospatial Data Abstraction Library) and provides a more user-friendly and Pythonic interface.

Here are some key features and functionalities of Rasterio:

* **Reading and Writing Raster Data:** Rasterio supports reading and writing various raster formats, including GeoTIFF, JPEG, PNG, and more. It provides an easy way to access the metadata, spatial reference system (SRS), and other properties of the raster dataset.

* **Data Manipulation:** Rasterio allows you to perform various operations on raster data, such as cropping, reprojecting, resampling, warping, and merging. It provides efficient memory-mapped access to the raster data, enabling processing of large datasets.

* **Georeferencing and Coordinate Transformation:** Rasterio handles the transformation between pixel coordinates and real-world geographic coordinates using the affine transformation matrix. It provides functionality to convert between different coordinate reference systems (CRS) and perform spatial transformations.

* **Masking and Clipping:** Rasterio provides tools to mask out specific areas of a raster using boolean masks or geometries. It allows you to clip rasters using bounding boxes, polygons, or other shapes.

* **Dataset Metadata and Attributes:** Rasterio allows you to access and modify the metadata and attributes associated with raster datasets, including band information, nodata values, and color mapping.

* **Parallel Processing:** Rasterio supports parallel processing of raster data using Python's multiprocessing module, allowing efficient utilization of multi-core systems for raster operations.

* **Integration with Geospatial Libraries:** Rasterio seamlessly integrates with other geospatial libraries such as NumPy, Matplotlib, GeoPandas, and Shapely, enabling powerful geospatial analysis workflows.

Overall, Rasterio provides a convenient and efficient way to work with raster data in Python, making it a valuable tool for geospatial analysis, remote sensing, GIS (Geographic Information System) applications, and other fields that deal with geospatial datasets.


## **01. Importing Required Libraries**

```python
import os
import rasterio
from rasterio.plot import show
import numpy as np
import geopandas as gpd
import shapely.geometry
import matplotlib.pyplot as plt
```

## **02. Setting Up the Current Working Directory**

```python
# Checking the current working directory
os.getcwd()
```

```python
# Changing the current working directory
file_path = r"D:\GIS Project\Raster Files"
os.chdir(file_path)
```

```python
# Checking the current working directory again
os.getcwd()
```

## **03. Reading the Raster File with Rasterio**


**Dataset Description:**<br>
The NASA Shuttle Radar Topography Mission (SRTM) 30m Digital Elevation Model (DEM) dataset provides high-resolution elevation data for the Earth's surface. This dataset is derived from radar measurements collected by the Space Shuttle Endeavour during a 2000 mission.

For a specific area, the SRTM 30m DEM dataset offers a detailed representation of the terrain, capturing elevation values at a resolution of approximately 30 meters. This level of detail allows for precise analysis and modeling of the topography, making it valuable for a range of applications, including hydrology, terrain analysis, visualization, and environmental studies.

```python
# Reading raster from local file with rasterion
elev = rasterio.open(file_path + "\\Bankura_SRTM_DEM.tif", mode="r")
```

```python
# Checking the datatype of the elev variable
type(elev)
```

## **04. Rasterio Attributes and Methods**

```python
# Checking the name/path of the raster file
elev.name
```

```python
# Checking the metadata of the raster file
elev.meta
```

```python
# Checking the driver, crs and count separately
print("Raster Driver:", elev.driver)
print("Raster CRS:", elev.crs)
print("Raster Count:", elev.count)
```

**bounds Method:** <br>
bounds method is used to retrieve the bounding box or extent of a raster dataset. The bounding box represents the minimum and maximum coordinates in the x and y dimensions that encompass the entire raster.

```python
# Checking the bounding box of the data
elev.bounds
```

By using the asterisk (*) before elev.bounds, it unpacks the values from elev.bounds and passes them as separate arguments to the shapely.geometry.box function.

```python
print(*elev.bounds)
```

**shapely.geometry.box:** <br> This method in Shapely is used to create a rectangular polygon, also known as a bounding box. It creates a Shapely geometry object representing a rectangular region defined by its minimum and maximum x and y coordinates.

```python
# Converting the bounding box into shapely geometry object
bbox = shapely.geometry.box(*elev.bounds)
# Converting the bounding box into geopandas geoseries object
bbox_geo = gpd.GeoSeries(bbox)
```

```python
# Plotting the bounding box
bbox_geo.plot()
plt.title("Bounding Box")
plt.xlabel("Longitude (DD)")
plt.ylabel("Latitude (DD)")
plt.show()
```

## **05. Creating a Subset Area**

```python
# Reading the Shapefile of the region of interest
file_path = r"D:\GIS Project\ShapeFiles\Bankura District\Bankura_District.shp"
roi = gpd.read_file(file_path)
```

```python
# Checking the first five rows of the geodataframe
roi.head()
```

```python
# Printing all the block names
print(roi["Block"].unique())
```

```python
# Extracting the Bankura I for subset area
roi_subset = roi["geometry"][roi["Block"]=="Bankura I"]
```

```python
# Plotting the roi_subset and bounding box
roi_subset.plot()
plt.title("ROI Subset")
plt.xlabel("Longitude (DD)")
plt.ylabel("Latitude (DD)")
plt.show()
```

```python
# Converting the Bankura I into shapely geometry
from shapely.geometry import Polygon
roi_subset_geometry = roi_subset.geometry.to_list()
print(roi_subset_geometry)
print(type(roi_subset_geometry))
# Getting the first item from the roi_subset_geometry list
roi_subset_poly = roi_subset_geometry[0]
roi_subset_poly
```

```python
# Checking the bounds of the roi_subset_poly
minx, miny, maxx, maxy = roi_subset_poly.bounds
```

## **06. Visualizing Raster Data**

```python
# Visualizing the whole DEM data
show(elev)
```

**from_bounds:** <br> This function is used to create a raster dataset that covers a specific bounding box in the desired CRS (coordinate reference system).
**transform:**<br>In Rasterio, the .transform attribute is used to access the affine transformation matrix of a raster dataset. The transformation matrix defines the spatial relationship between the pixel coordinates of the raster and its real-world coordinates.

```python
# Visualizing DEM data of the roi_subset area
from rasterio.windows import from_bounds
raster_path = r"D:\GIS Project\Raster Files\Bankura_SRTM_DEM.tif"
with rasterio.open(raster_path, mode="r") as src:
    dem = src.read(1, window=from_bounds(minx, miny, maxx, maxy, src.transform))
    plt.title("ROI Subset DEM")
    show(dem)
```

```python
# Visualizing DEM data of the roi_subset area in grayscale
plt.title("ROI Subset DEM")
show(dem, cmap="gray")
```

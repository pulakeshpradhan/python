[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Image-Analysis-in-Remote-Sensing-with-Python/01_Reading_and_Displaying_Image_Band.ipynb)

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

# **Reading and Displaying Rasters Using GDAL in Python**
**Author: Krishnagopal Halder** <br>

GDAL (Geospatial Data Abstraction Library) is a powerful open-source library for reading, writing, and manipulating geospatial raster and vector data formats. It provides a wide range of functionalities to work with various GIS (Geographic Information System) data formats, including popular ones like GeoTIFF, Shapefile, and many others. In Python, GDAL can be accessed through the osgeo module, which is a part of the GDAL project.

GDAL in Python allows you to perform tasks such as reading and writing raster and vector datasets, extracting metadata, transforming and projecting data, and performing various geospatial analysis operations. With its extensive capabilities, GDAL enables developers and data scientists to process geospatial data efficiently, making it a valuable tool in fields such as remote sensing, environmental modeling, and geospatial analysis.

To work with GDAL in Python, you'll need to have the GDAL library installed on your system. You can install it using package managers like pip or conda. Once installed, you can import the necessary modules from osgeo to begin working with geospatial data.


## **01. Importing Required Libraries**

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from osgeo import osr
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
file_path = r"D:\GIS Project\Research Projects\Mangrove_Extent_Mapping\New Satellite Images"
os.chdir(file_path)
```

```python
# Checking the current working directory
os.getcwd()
```

## **03. Reading Raster with GDAL**


**gdal.AllRegister():** <br>
gdal.AllRegister() is a function from the GDAL (Geospatial Data Abstraction Library) library. It is used to register all available GDAL drivers. Before using any GDAL functionality, it is necessary to register the available drivers. The gdal.AllRegister() function ensures that all GDAL drivers are registered and available for use in the current session. By registering the drivers, GDAL becomes aware of the formats it can read and write, allowing you to work with different geospatial data formats seamlessly.

```python
# Registering all available GDAL drivers
gdal.AllRegister()
```

**Dataset Description:** <br>
The raster data that has been created using the Google Earth Engine platform by compositing cloud masked Landsat 8 images of the year 2022 is a multi-band dataset comprising a total of 9 bands. Each band represents a different type of spectral index derived from the Landsat 8 satellite imagery.

Spectral indices are mathematical calculations applied to remote sensing data to extract specific information about the land surface characteristics. These indices are often used to monitor vegetation health, detect changes in land cover, assess water quality, and perform various other types of analysis.

```python
# Defining the path of the raster image
raster_name = "Mangrove_Composite_2022_New.tif"
file = file_path + "\\" + raster_name
```

```python
# Checking the metadata of the raster image using command line
!gdalinfo -nomd Mangrove_Composite_2022_New.tif
```

```python
# Reading the raster dataset with GDAL
dataset = gdal.Open(file, GA_ReadOnly)
```

## **04. Checking the Properties of the Raster File**

```python
# Checking the type of the dataset variable
type(dataset)
```

RasterXSize, RasterYSize, and RasterCount are attributes of a GDAL dataset object. These attributes provide information about the size and composition of a raster image.

* **RasterXSize:** This attribute represents the width of the raster image in pixels. It provides the number of columns or the horizontal dimension of the image.

* **RasterYSize:** This attribute represents the height of the raster image in pixels. It provides the number of rows or the vertical dimension of the image.

* **RasterCount:** This attribute represents the number of bands or layers in the raster image. A band typically represents a specific data type or characteristic of the image, such as red, green, blue (RGB) bands in a color image or different spectral bands in a multispectral image. Each band contains pixel values corresponding to its specific data type or characteristic.

```python
# Getting the width, height, number of bands of the image
nCols = dataset.RasterXSize
nRows = dataset.RasterYSize
nBands = dataset.RasterCount
print("Number of Cloumns (Image Width):", nCols)
print("Number of Rows (Image Height):", nRows)
print("Number of Bands:", nBands)
```

GetProjection() and GetGeoTransform() are methods of a GDAL dataset object. These methods provide information about the spatial properties and coordinate system of a raster image.

* **GetProjection():** This method retrieves the projection information of the raster image. The projection defines how the geographic coordinates (latitude and longitude) are mapped to the pixel coordinates in the image. The projection information is typically represented as a well-known text (WKT) string, which describes the coordinate reference system (CRS) used by the image.

* **GetGeoTransform():** This method retrieves the georeferencing information of the raster image. The georeferencing information describes the transformation between the pixel coordinates in the image and real-world geographic coordinates. It consists of six coefficients that define the origin, pixel size, and rotation of the image.

```python
# Getting the coordinate information of the raster image
print("Coordinate System:", dataset.GetProjection(), sep="\n")
```

```python
# Getting the transformation coefficients
geoTransform = dataset.GetGeoTransform()
print(geoTransform)
```

```python
print("Origin:", geoTransform[0], geoTransform[3])
print("Pixel Size:", geoTransform[1], geoTransform[5])
```

```python
print("Upper Left Corner:", gdal.ApplyGeoTransform(geoTransform, 0, 0))
print("Upper Right Corner:", gdal.ApplyGeoTransform(geoTransform, nCols, 0))
print("Lower Left Corner:", gdal.ApplyGeoTransform(geoTransform, 0, nRows))
print("Lower Right Corner:", gdal.ApplyGeoTransform(geoTransform, nCols, nRows))
print("Center:", gdal.ApplyGeoTransform(geoTransform, nCols/2, nRows/2))
```

GetMetadata() is a methods of a GDAL dataset object. This method is used to retrieve metadata associated with a raster image.

* **GetMetadata():** This method retrieves all metadata associated with the dataset. Metadata provides additional information about the dataset, such as the data source, acquisition parameters, processing history, or any other relevant details. The GetMetadata method returns a dictionary object containing key-value pairs of the metadata items.

* **GetMetadata("IMAGE_STRUCTURE"):** This method retrieves metadata specifically related to the image structure. The "IMAGE_STRUCTURE" parameter is passed to specify the category of metadata to retrieve. This category typically contains information about the structure and organization of the image data, such as the color interpretation, pixel data type, compression, or block size.

```python
# Checking the metadata of the raster
print("Metadata:", dataset.GetMetadata())
```

```python
print("Image Structure Metadata:", dataset.GetMetadata("IMAGE_STRUCTURE"))
```

## **05. Reading the Bands of the Raster**


**GetRasterBand():** is used to retrieve each band of the dataset within a loop. The band index starts from 1, so nBands + 1 is passed as an argument to GetRasterBand() to obtain each band.

The **GetDescription()** method is then called on each band object (band) to retrieve the band description. The band description can be a user-defined label or a description assigned to the band.

After that, the **ComputeStatistics()** method is used to compute statistics for each band. The False argument passed to ComputeStatistics() indicates that the function should not force computation of statistics if they are not available.

```python
# Extracting the description of all the bands
for i in range(1, nBands + 1):
    band = dataset.GetRasterBand(i)
    band_description = band.GetDescription()
    print(f"Band{i}: {band_description}")
```

```python
# Checking statistics of the raster image using command line
!gdalinfo -stats Mangrove_Composite_2022_New.tif
```

```python
# Calculating the statistics of all the bands
for i in range(1, nBands + 1):
    band = dataset.GetRasterBand(i)
    (minimum, maximum, mean, stdDev) = band.ComputeStatistics(False)
    print("band{:d}, min={:.3f}, max={:.3f}, mean={:.3f}, stdDev={:.3f}"\
          .format(i, minimum, maximum, mean, stdDev))
```

The **ReadAsArray()** method in GDAL is used to read the pixel values of a raster band into a NumPy array. It allows you to access the actual pixel data for further processing or analysis.

```python
# Reading the first band (NDVI) of the raster image
ndvi = dataset.GetRasterBand(1)
# Converting the raster band into NumPy array
ndviArray = ndvi.ReadAsArray()
```

```python
# Printing the NDVI array
ndviArray
```

```python
# Printing the shape of the NDVI array
ndviArray.shape
```

```python
# Reading the second band of the raster image
ndwi = dataset.GetRasterBand(2)
# Converting the raster band into NumPy array
ndwiArray = ndwi.ReadAsArray()
```

```python
# Checking the shape of the NDWI array
ndwiArray.shape
```

## **06. Plotting the Raster Band**

```python
# Plotting the NDVI band
plt.figure(figsize=(8, 6))
plt.imshow(ndviArray, cmap="RdYlGn")
plt.title("Normalized Difference Vegetation Index (NDVI)")
plt.xlabel("Image X Coordinate")
plt.ylabel("Image Y Coordinate")
plt.colorbar(orientation="horizontal")
plt.show()
```

```python
plt.figure(figsize=(8, 6))
plt.imshow(ndwiArray, cmap="YlGnBu")
plt.title("Normalized Difference Water Index (NDWI)")
plt.xlabel("Image X Coordinate")
plt.ylabel("Image Y Coordinate")
plt.colorbar(orientation="horizontal")
plt.show()
```

## **07. Automating the Process using Custom Function**

```python
# Creating a function to check the bands description of a raster
def checkBands(data):
    gdal.AllRegister()
    dataset = gdal.Open(data, GA_ReadOnly)
    
    # Getting the number of bands in the raster
    nBands = dataset.RasterCount
    
    # Printing the raster band description
    for i in range(1, nBands + 1):
        band = dataset.GetRasterBand(i)
        band_info = band.GetDescription()
        print(f"Band{i}: {band_info}")
```

```python
# Creating a function to plot the bands of a raster
def plotBand(data, band_num, palette, title):
    # Reading the raster band
    gdal.AllRegister()
    dataset = gdal.Open(data, GA_ReadOnly)
    rasterBand = dataset.GetRasterBand(band_num)
    
    # Convering the raster image to NumPy array
    bandArray = rasterBand.ReadAsArray()
    
    # Plotting the raster band
    plt.figure(figsize=(8, 6))
    plt.imshow(bandArray, cmap=palette)
    plt.title(title)
    plt.xlabel("Image X Coordinate")
    plt.ylabel("Image Y Coordinate")
    plt.colorbar(orientation="horizontal")
    plt.show()
```

```python
# Testing the functions
data_path = r"D:\GIS Project\Research Projects\Mangrove_Extent_Mapping\New Satellite Images\Mangrove_Composite_2022_New.tif"
# Checking the bands information using checkbands() function
checkBands(data_path)
```

```python
# Plotting the bands using the plotBand() function
plotBand(data_path, 7, "YlGn", "Soil Adjusted Vegetation Index (SAVI)")
```

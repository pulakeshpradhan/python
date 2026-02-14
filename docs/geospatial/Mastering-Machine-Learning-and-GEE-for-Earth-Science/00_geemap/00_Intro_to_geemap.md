---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/00_geemap/00_Intro_to_geemap.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="dU4mDT2ygAOd" -->
# **Intro to geemap**
<img src="https://geemap.org/assets/logo.png" width="20%">
<!-- #endregion -->

<!-- #region id="nV_6mhbAfK7H" -->
## **Installing the Updated Version of geemap**
Install geemap version `0.29.3` or a later release to enable the automatic authentication feature, as this functionality is only supported in these versions. To install other Python packages, you can use the `pip install package_name` command. To update a package, you can use `pip install --upgrade package_name` or `pip install -U package_name`.
<!-- #endregion -->

```python id="3Nu69rwRf2ZU"
# %pip install -U geemap
```

<!-- #region id="xiAw2gqigGT_" -->
## **Import the Required Libraries**
<!-- #endregion -->

```python id="TAvvfdPugQCg"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ee
import geemap
```

<!-- #region id="conlGhpFipp3" -->
## **Initialize a Map**

When initializing a Map object, you may be prompted for authorization. If this occurs, you can obtain the required authorization token by visiting the provided link.

Certainly! When working with the Map object in a Python environment, you can play with various parameters to customize the map display.
<!-- #endregion -->

```python id="fAyo0CkEiuyY"
Map = geemap.Map()
Map
```

```python id="jbqnWaFonxPa"
# Change the map height and width parameter
Map = geemap.Map(height="400pt", width="800pt")
Map
```

```python id="mLurHUdaqBXr"
# Print the list of basemaps
basemaps = geemap.basemaps

for basemap in basemaps:
    print(basemap)
```

```python id="DMNjQgfnpZ0F"
# Change the basemap layer to 'Esri World Imagery'
Map.add_basemap(basemap="Esri.WorldImagery")
```

<!-- #region id="A-zlyNBUptfH" -->
## **Working with Feature Collection**
In Google Earth Engine, a Feature Collection is a type of data structure that represents a collection of vector features. These features could represent points, lines, polygons, or a combination of these geometries. Feature Collections are fundamental for working with spatial data and conducting geospatial analyses in Earth Engine.

<img src="https://miro.medium.com/v2/resize:fit:1284/1*TaID5vtnzOYkkKSb-VQNpg.png" width="45%">
<!-- #endregion -->

```python id="seisMMFQq14u"
# Set up a new map object
Map = geemap.Map(height="400pt", width="100%")
Map
```

```python id="1HTb8wvhrHTQ"
# Import 'World Administrative Boundary' shapefile layer as a feature collection
world = ee.FeatureCollection("users/geonextgis/World_Administrative_Boundaries")

# Set visualization/style parameters
world_style = {
    "fillColor": "00000000", # transparent color code
    "color": "black", # color of the stroke
    "width": 0.5 # stroke width
}

# Display the layer
Map.addLayer(world.style(**world_style), {}, "World Administrative Boundaries")
```

<!-- #region id="-YKr7Nx1xCa_" -->
### **Filtering Feature Collection:**
Filtering a Feature Collection in Google Earth Engine involves selecting a subset of features based on specific criteria, such as spatial, attribute, or temporal conditions. This process is essential for isolating relevant data for analysis. Here are the key aspects of filtering a Feature Collection:
<!-- #endregion -->

<!-- #region id="vHg967Ed42xN" -->
1. **Attribute Filtering:**

    - Attribute filtering involves selecting features based on their attribute values, such as properties or characteristics.
    - The `filter` function is often used in combination with `ee.Filter` to define attribute-based conditions.
<!-- #endregion -->

```python id="p-6_3pxI5I-d"
# Filter all the Asian countries
asian_countries = world.filter(ee.Filter.eq("continent", "Asia"))
asian_countries_style = {
    "fillColor": "93939388",
    "color": "black",
    "width": 1
}
Map.addLayer(asian_countries.style(**asian_countries_style), {}, "Asian Countries")
Map.centerObject(asian_countries, 3)
```

<!-- #region id="h29pScitxlE6" -->
2. **Spatial Filtering:**
   - Spatial filtering involves selecting features based on their geographic location or proximity to a specified region.
   - Common spatial filters include `geometry`, `intersects`, `bounds`, and `distance`, allowing users to focus on features within a defined area or at a certain distance from a given point.
<!-- #endregion -->

```python id="7yos-gPEv4jw"
# Read 'Gridded Population of the World Version 4' point dataset provided by NASA SEDAC
gpw = ee.FeatureCollection("projects/sat-io/open-datasets/sedac/gpw-v4-admin-unit-center-points-population-estimates-rev11")

# Filter the feature collection with only Asian countries (Spatial Filtering)
# Filter the points where population estimates is more than 5000000 in 2020 (Attribute Filtering)
gpw_asia = gpw.filterBounds(asian_countries)\
              .filter(ee.Filter.gt("UN_2020_E", 5000000))

gpw_asia_style = {
    "fillColor": "C70039",
    "color": "black",
    "width": 1,
    "pointSize": 5
}
Map.addLayer(gpw_asia.style(**gpw_asia_style), {}, "Global Population Estimates > 5 Lakhs (Asia)")
```

<!-- #region id="oPEJVNFVyrKY" -->
### **Downloading Feature Collection:**

 🤔 **Note:** It's always a good practice to comment out the `task.start()` line when sharing code to avoid unintentional multiple downloads of the same file.
<!-- #endregion -->

```python id="myOwTbnCyyaA"
# Initialize an export task
task = ee.batch.Export.table.toDrive(collection=gpw_asia,
                                     description="GPW_Asia_Pop_Est_2020",
                                     folder="GEE",
                                     fileNamePrefix="GPW_Asia_Pop_Est_2020",
                                     fileFormat="SHP")

# Export the filtered feature collection
# task.start()
```

<!-- #region id="dqiPxq5ZB-K7" -->
🤔 **Note:** In geemap, the addLayer function is designed to visualize data on the map by adding a layer. However, it's important to note that this function always returns an Image object, not a Feature Collection or individual Feature when using with `style` function.

🔑 **Exercise:**
 - Filter African countries from the `World Administrative Boundary` layer.
 - Filter global population data within the selected African countries.
 - Identify points where population estimates are more than 500,000 (5 lakhs) in 2020.
 - Visualize the filtered layers on the map with custom styling.
 - Download the filtered layers into the Google drive.
<!-- #endregion -->

<!-- #region id="RHMtRcAzyPyM" -->
## **Working with Image Collection**
In Google Earth Engine (GEE), an Image Collection is a fundamental data structure used to represent a group or sequence of images. These images can be satellite observations, remotely sensed data, or any other raster data that can be organized over time or space.

<img src="https://www.mdpi.com/remotesensing/remotesensing-14-02778/article_deploy/html/images/remotesensing-14-02778-g001.png">
<!-- #endregion -->

```python id="yN0bzy4T0DYB"
# Set up a new map object
Map = geemap.Map(height="400pt", width="100%")
Map
```

```python id="l423BLL41jTM"
# Add a marker to the map and convert it into an EE feature
marker = Map.draw_last_feature
# marker
```

```python id="LEzEwbuD7wAO"
# Import USGS Landsat 9 Level 2, Collection 2, Tier 1 image collection
L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
```

<!-- #region id="Ni6TGzlr6xRW" -->
### **Filtering Image Collection**
  - We can filter Image Collections based on various criteria, such as `date range`, `spatial extent`, or `metadata` properties.
  - To optimize the workflow, it is advisable to follow a specific order when filtering an Image Collection. The recommended sequence includes filtering by boundary first, followed by dates, and then metadata properties.
  - This approach helps reduce computational load and speeds up the execution of operations.
<!-- #endregion -->

```python id="0ZQQXlh08o8r"
# Filter the Landsat 9 image collection using marker points (Filter with boundary)
# Select images acquired in the year 2022 (Filter with date range)
# Filter the collection to include images with less than 10% cloud cover (Filter with metadata)
# Choose the first image from the filtered image collection

L9Filtered = L9.filterBounds(marker.geometry())\
               .filterDate("2022-01-01", "2022-12-31")\
               .filterMetadata("CLOUD_COVER", "less_than", 10)\
               .first()

# Display the RGB and SFCC images into the map
rgb_vis = {
    "min": 8000,
    "max": 17000,
    "bands": ["SR_B4", "SR_B3", "SR_B2"]
}

sfcc_vis = {
    "min": 8000,
    "max": 17000,
    "bands": ["SR_B5", "SR_B4", "SR_B3"]
}

Map.addLayer(L9Filtered, rgb_vis, "RGB Composite")
Map.addLayer(L9Filtered, sfcc_vis, "SFCC Composite")
Map.centerObject(marker, 9)
```

<!-- #region id="-G8nSm5rBfDl" -->
🤔 **Note:** `getInfo()` is a method in GEE API that allows users to retrieve the values of Earth Engine objects and transfer them from the server-side to the client-side. In the context of GEE, computations often occur on the server-side, which means that the actual data and results reside on Google's servers. To access and work with this information in your local environment, we use `getInfo()`.
<!-- #endregion -->

```python id="OFF4tJTj_huA"
# Store the metadata property names in a list
image_prop_names = L9Filtered.propertyNames()

# Print the properties information
image_props = L9Filtered.toDictionary(image_prop_names).getInfo()
image_props
```

<!-- #region id="lEhpm_AYCcnr" -->
🔑 **Exercise:**
 - Load Landsat 9 image collection for a specific region over a multi-year period (2020-2022).
 - Filter the image collection based on the 'CLOUD_COVER' property (< 5%).
 - Visualize the image in three different band combinations, such as (Red, Green, Blue), (NIR, Red, Green), and (SWIR2, NIR, Red).
 - Print metadata properties of the image.
<!-- #endregion -->

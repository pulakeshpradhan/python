[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Image-Analysis-in-Remote-Sensing-with-Python/Machine Learning with Scikit Learn/Download_Landsat_8_Data.ipynb)

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

## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ee
import geemap
```

## **Set Up Earth Engine**

```python
# Authenticating and initializing Earth Engine API
# ee.Authenticate()
```

```python
# ee.Initialize()
```

```python
# Creating a Map object
m = geemap.Map(width="100%", height="550px")
```

## **Add the Region of Interest**

```python
# Define the region of interest
roi = ee.Geometry.Polygon([[88.282386, 22.519994], 
                           [88.391925, 22.519994],
                           [88.391925, 22.623027],
                           [88.282386, 22.623027],
                           [88.282386, 22.519994]])
# Display the roi
m.addLayer(roi, {}, "Region of Interest")
m.centerObject(roi, 12)
m
```

## **Prepare the Data**

```python
# Import Landsat 8 imagery
l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

# Filter the image collection
l8Filtered = l8.filterBounds(roi)\
               .filterDate("2022-01-01", "2022-12-31")\
               .filterMetadata("CLOUD_COVER", "less_than", 10)\
               .select("SR_B.*")

# Create a median composite
composite = l8Filtered.median()\
                      .clip(roi)
```

```python
# Create a function to apply scale factor
def applyScaleFactor(image):
    return image.multiply(0.0000275).add(-0.2)

# Apply the function over the composite image
scaled_composite = applyScaleFactor(composite)
```

```python
# Display the selected image
rgbVis = {
    "min": 0.0,
    "max": 0.3,
    "bands": ["SR_B5", "SR_B4", "SR_B3"],
}
m.addLayer(scaled_composite, rgbVis, "Landsat 8 Composite Image")
m
```

```python
# Create a function to add Indices
def addIndices(image):
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    mndwi = image.normalizedDifference(["SR_B3", "SR_B6"]).rename("MNDWI")
    ndbi = image.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
    savi = image.select("SR_B5").subtract(image.select("SR_B4"))\
                .divide(image.select("SR_B5").add(image.select("SR_B4").add(0.5)))\
                .multiply(1.5).rename("SAVI")
    
    return image.addBands(ndvi)\
                .addBands(mndwi)\
                .addBands(ndbi)\
                .addBands(savi)

# Apply the addIndices function over the scaled image
withIndices = addIndices(scaled_composite)
m.addLayer(withIndices, rgbVis, "Landsat 8 with Indices")
m
```

## **Export the Imagery to Drive**

```python
# Create a task to export the image
task = ee.batch.Export.image.toDrive(image=withIndices.toDouble(),
                                     description='Landsat_8_Image',
                                     folder="GEE",
                                     fileNamePrefix="Landsat_8_Image_Kolkata",
                                     region=roi,
                                     scale=30,
                                     maxPixels=1e10,
                                     fileFormat="GeoTIFF")
```

```python
# Start the task
# task.start()
```

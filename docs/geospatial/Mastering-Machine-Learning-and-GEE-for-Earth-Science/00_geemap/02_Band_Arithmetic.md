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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/00_geemap/02_Band_Arithmetic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="6-7J5immvJK9" -->
# **Band Arithmetic**
Band arithmetic in remote sensing refers to mathematical operations or calculations performed on the digital values of individual bands of remotely sensed satellite or aerial imagery.

Two important functions in GEE are `normalizedDifference` and `expression`, which are commonly used for image processing and analysis.

1. **`normalizedDifference` Function:**
   - The `normalizedDifference` function in GEE is used to compute the normalized difference between two bands of a satellite image. Normalized differences are often used in remote sensing to highlight specific features or characteristics of the Earth's surface.
   - Syntax: `normalizedDifference([bandName1, bandName2])`
   - Example:
     ```python
        NDVI = image.normalizedDifference(['NIR', 'Red'])
     ```
   - In this example, NDVI (Normalized Difference Vegetation Index) is calculated by taking the normalized difference between the Near-Infrared (NIR) and Red bands of a satellite image. NDVI is commonly used to assess vegetation health and density.

   - **When to Use:**
    - Use `normalizedDifference` when you want to compute standard normalized difference indices such as NDVI, NDWI, NDBI (Normalized Difference Built-Up Index), etc.
    - It is a convenient function for simple band arithmetic that follows a standard formula.


2. **`expression` Function:**
   - The `expression` function allows users to create custom mathematical expressions to manipulate and combine bands within an image. It provides flexibility for performing complex operations on remote sensing data.
   - Syntax: `expression(expressionString, parameters)`
   - Example:
     ```python
        NDWI = image.expression('(GREEN - NIR) / (GREEN + NIR)', {
            'GREEN': image.select('GREEN'),
            'NIR': image.select('NIR')
        })
     ```
   - In this example, the NDWI (Normalized Difference Water Index) is calculated using a custom expression that involves the Green and NIR bands. NDWI is commonly used for water detection and monitoring.

   - **When to Use:**
    - Use `expression` when you need to create custom mathematical expressions that go beyond simple normalized differences. This is useful for complex band combinations or mathematical operations.
    - It allows you to define your own index or perform advanced operations on bands, providing more flexibility and control over the image processing workflow.
<!-- #endregion -->

<!-- #region id="RchKugASxtMI" -->
## **Import the Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="jtCpeK0V3Dcx" outputId="ed64d482-3f97-4f77-bbc0-e23609ae455a"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="LSZlAfcbxddN" colab={"base_uri": "https://localhost:8080/", "height": 17} outputId="f9371daf-5165-4923-b74c-eb1ecb8b9322"
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import ee
import geemap
```

<!-- #region id="EzVmXv7bxzUr" -->
## **Initialize a Map**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="Cp8Jph4818M3" outputId="e73da2da-e08a-48a3-cc9e-3419e1be0216"
# # Trigger the authentication flow.
# ee.Authenticate()

# # Initialize the library.
# ee.Initialize(project='my-project')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 554, "referenced_widgets": ["8d07c54aca0347ad86203d2ee52f2054", "82a6a0f9d55d4358a4c8e0d68a15d285", "37506c4546894f3dbac0cf61512325ad", "43424ecf4a6c4f5c87b8922329b910c3", "1e1c64f413bb45f394def9f539928686", "09fc2877d4a04c4bab96f4b451e6c202", "cf91367838484eabb72bb31f68358761", "3d420d04c0db41a7aead6410d27d341c", "3ba332ada2644ed29d6e755c6f2c97c2", "c6d3110caf43478c88d9391518ed7284", "f37140036d444249864088a6cdf3de5c", "41c2c26076fc47f4a302c7e653752cd3", "6cc08b431e0f48ad9c9a6a6bef8ab10d", "0d7e3ca56a5c408da31cd04a5fff26fa", "961ed37951474fcf988498169465fafc", "cba218ba97214212b347310558afe558", "dc96e908d0824aa38c7388614ea7a158", "3f0877c6cfb74130ab363d11d1e00655", "d3ea7caa24014fcaa46b3c00ac8ad5fc", "7598abc0457c45d2b02d8b9978fd3d43", "ea3b45e62b384490be7750167e03c2e9", "e7d40bfa50e74e7d955c85fa5c264c25", "17af8fa808c74e52bbc5102cd6e837f9", "0a9c8cd44ced4a0da547ca1c715d3163", "5a98f9da9d324ab48d94282b8c265d6f", "8787a141a43b4758a5a582aff241a849", "f86ccddca5b24173a57bee8dc4c906e3", "05b696e99ab64ed999c5098484d96a94"]} id="0txWpZFG1yz3" outputId="84251dcf-a870-43ef-d14a-8331dba553b4"
Map = geemap.Map(height="400pt")
Map
```

<!-- #region id="ncLmgxwd2gh9" -->
## **Define a Region of Interest**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 223} id="t4XvkoER3mx8" outputId="c60e72c4-9577-4484-8397-0b8cc9c0da65"
# Read the shapefile of the West Bengal state using geopandas
shp_path = r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/West_Bengal_Boundary/District_shape_West_Bengal.shp"
wb_gdf = gpd.read_file(shp_path)
print(wb_gdf.shape)
wb_gdf.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 89} id="py8GOFDG4Vcx" outputId="e7771025-05c1-4df5-b1a6-e867e702a784"
# Filter the 'Bankura' district from the geodataframe
roi_gdf = wb_gdf[wb_gdf['NAME']=="Bankura"]
roi_gdf
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="q1bvvzFT4tt1" outputId="ec3da44e-0285-4693-ca5e-9a5beda22947"
# Push the filtered geometry to the Earth Engine
roi_ee = geemap.gdf_to_ee(roi_gdf)

vis_params = {
    "fillColor": "00000000",
    "color": "black",
    "width": 1
}
Map.addLayer(roi_ee.style(**vis_params), {}, "ROI")
Map.centerObject(roi_ee, 9)
```

<!-- #region id="h3Jvfz7K7K7W" -->
## **Band Arithmetic on Landsat Data**
<!-- #endregion -->

<!-- #region id="KMvvsJdxHTnz" -->
### **Filtering Landsat 9 Image Collection**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="TApngSNF7Uze" outputId="04a1af13-416a-4e7f-c11b-5aafd2999e41"
# Read Landsat 9 image collection from EE
L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

# Filter the image collection with roi, daterange, and cloud cover property
L9Filtered = L9.filterBounds(roi_ee)\
               .filterDate("2022-01-01", "2022-12-31")\
               .filterMetadata("CLOUD_COVER", "less_than", 50)

# Print the size of the filtered image collection
L9Filtered.size().getInfo()
```

<!-- #region id="K8Y9MoUUHZqy" -->
### **Preprocessing on Landsat Data**
<!-- #endregion -->

```python id="od1urzmGHKAz" colab={"base_uri": "https://localhost:8080/", "height": 17} outputId="220d4701-c910-4218-ef35-04aaca5d2015"
# Write a function to rename Landsat 9 band names
def renameL9(image):
    # Define the existing band names
    band_names = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6',
                  'SR_B7', 'SR_QA_AEROSOL', 'ST_B10', 'ST_ATRAN', 'ST_CDIST', 'ST_DRAD',
                  'ST_EMIS', 'ST_EMSD', 'ST_QA', 'ST_TRAD', 'ST_URAD', 'QA_PIXEL',
                  'QA_RADSAT']

    # Define the new band names
    new_band_names = ['COASTAL', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1',
                      'SWIR2', 'AEROSOL', 'THERMAL', 'ST_ATRAN', 'ST_CDIST', 'ST_DRAD',
                      'ST_EMIS', 'ST_EMSD', 'ST_QA', 'ST_TRAD', 'ST_URAD', 'QA_PIXEL',
                      'QA_RADSAT']

    # Rename the band names
    return image.rename(new_band_names)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="gxVsL_S98Tx-" outputId="7ba64f34-09cc-4aba-eb96-98b0fc497940"
# Write a function to remove clouds from Landsat 9 imagery
def maskL9CloudsAndShadows(image):

    # Read the 'QA_PIXEL' (Quality Assessment) band
    qa = image.select("QA_PIXEL")

    # Define all the variables
    dilated_cloud_bitmask = 1 << 1
    cirrus_bitmask = 1 << 2
    cloud_bitmask = 1 << 3
    cloud_shadow_bitmask = 1 << 4

    # Create a mask
    mask = qa.bitwiseAnd(dilated_cloud_bitmask).eq(0).And(
           qa.bitwiseAnd(cirrus_bitmask).eq(0)).And(
           qa.bitwiseAnd(cloud_bitmask).eq(0)).And(
           qa.bitwiseAnd(cloud_shadow_bitmask).eq(0))

    return image.updateMask(mask)
```

<!-- #region id="u2M0NyWQsHyE" -->
### **Function to Calculate Various Spectral Indices**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="bfTmcvEosbpI" outputId="5cd0e819-6dac-4c66-98a5-2c9788e68b95"
# Write a function to calculate NDVI, NDWI, MNDWI, NDBI, BU, SAVI, EVI, GCVI of an image
def calculateIndices(image):

    # NDVI = (NIR - RED) / (NIR + RED)
    NDVI = image.normalizedDifference(["NIR", "RED"])\
                .rename("NDVI")

    # NDWI = (NIR – SWIR1) / (NIR + SWIR1)
    NDWI = image.normalizedDifference(["NIR", "SWIR1"])\
                .rename("NDWI")

    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    MNDWI = image.normalizedDifference(["GREEN", "SWIR1"])\
                 .rename("MNDWI")

    # NDBI = (SWIR – NIR) / (SWIR + NIR)
    NDBI = image.normalizedDifference(["SWIR1", "NIR"])\
                .rename("NDBI")

    # BU = NDBI - NDVI
    BU = NDBI.subtract(NDVI)\
             .rename("BU")

    # SAVI = ((NIR – RED) / (NIR + RED + 0.5)) * (1.5)
    SAVI = image.expression(
        "((NIR - RED) / (NIR + RED + 0.5)) * (1.5)", {
            "NIR": image.select("NIR"),
            "RED": image.select("RED")
    }).rename("SAVI")

    # EVI = 2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))
    EVI = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {
            "NIR": image.select("NIR"),
            "RED": image.select("RED"),
            "BLUE": image.select("BLUE")
    }).rename("EVI")

    # GCVI = (NIR/GREEN) − 1
    GCVI = image.expression(
        "(NIR / GREEN) - 1", {
            "NIR": image.select("NIR"),
            "GREEN": image.select("GREEN")
    }).rename("GCVI")

    # Add all the indices in a single ee list
    final_image = ee.Image([NDVI, NDWI, MNDWI, NDBI, BU, SAVI, EVI, GCVI])\
                    .copyProperties(image, ["system:time_start"])

    return ee.Image(final_image)
```

<!-- #region id="iforZ9iNHfvE" -->
### **Implementation on an Image and Image Collection**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="ipP7tUXVFrQJ" outputId="98b7bf6f-668d-476b-8c82-78d9f34348c3"
# Check the cloud cover value of the first image
# L9Filtered.first()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="mmYyHq2jF8Dq" outputId="91ab0b8a-5895-401f-eeb2-0c6aee5e3db6"
# Display the first image of the filtered image collection
L9FilteredFirst = L9Filtered.first()

# Apply 'renameL9', 'maskL9CloudsAndShadows', and 'calculateIndices' function on the image
renamedL9 = renameL9(L9FilteredFirst)
cloudMaksedL9 = maskL9CloudsAndShadows(renamedL9)
indicesL9 = calculateIndices(cloudMaksedL9)

# Display the indices image
Map.addLayer(indicesL9, {}, "L9 Spectral Indices")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="WXV4ze97_-aU" outputId="af4bb4c5-8f3c-4d32-e6f3-61cb40242015"
# Display the clipped NDVI image
NDVI_image = indicesL9.select("NDVI")\
                      .clip(roi_ee)

NDVI_vis = {
    'min': -0.064,
    'max': 0.332,
    'palette': ['#a50026', ' #da372a', ' #f67b4a', ' #fdbf6f', ' #feeea2',
                '#eaf6a2', ' #b7e075', ' #74c365', ' #229c52', ' #006837']
}

Map.addLayer(NDVI_image, NDVI_vis, "L9 NDVI Image")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="NIPsgMeXHo49" outputId="59dccde6-bf11-4253-f3ad-a52a2674a15c"
# Apply 'renameL9', 'maskL9CloudsAndShadows', and 'calculateIndices' function on the whole image collection
# to create a cloud free median composite spectral indices image of the whole year
medianIndicesL9= L9Filtered.map(renameL9)\
                           .map(maskL9CloudsAndShadows)\
                           .map(calculateIndices)\
                           .median()\
                           .clip(roi_ee)

Map.addLayer(medianIndicesL9, {}, "L9 Median Spectral Indices")
```

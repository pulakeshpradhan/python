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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/00_geemap/03_Download_Image_and_Image_Tiles.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="5gy3HeJpZDW5" -->
# **Download Image and Image Tiles from Earth Engine**
<!-- #endregion -->

<!-- #region id="00tE6aU7AKq6" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MkEGTunYAxLJ" outputId="d82da669-5e60-4239-c089-2452f660830e"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="uKcYkfjABQhN"
# %pip install rasterio
# %pip install geedim
```

```python id="k7zrKDm2A2wK" colab={"base_uri": "https://localhost:8080/", "height": 17} outputId="e86f28de-e075-47c1-919f-a952ae81d17a"
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import ee
import geemap
import rasterio
import geedim
```

<!-- #region id="aKbJP6cuBbnX" -->
## **Initialize a Map**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="g-pM8eCLBg2-" outputId="fb37512d-86ba-45af-9069-cd447ed4dfff"
# # Trigger the authentication flow.
# ee.Authenticate()

# # Initialize the library.
# ee.Initialize(project='my-project')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 554, "referenced_widgets": ["dcc41a0729ee47c78a13bf879d2afa47", "bb4b7e08c1424a54af96b3dee23c0714", "a4fa12c78fcc4541b966d66d9bf97cb8", "c636759c0a5c45f789585174673ee4d4", "e9fcdabd48524b94bf66925e9fc48a3b", "d6c446c84fe848ae9c72d32795b114fb", "065a8c140b9e4383a79cfa1a67dfc739", "3adb3d922e274aa5bcb4289ac0e77122", "8ad23638fa7445b9b6e50430236e89d1", "43f20132ce784cdbaa84287eea4343a6", "87bf7925f0664ed79d67ec68fd0ff562", "05e83c8f59e8498a8a045f46b2776b2b", "88e77d0aa1394110ad5f62e61a898c22", "ecf856e512264e54aa97d70527f192ff", "3939fde103394f5f9028e8d9b3eeee60", "dc0b058cb0914a75b5078895efaec11a", "829595534c0d4c20b511d475bda6d3f8", "057d2f340b3642a28f5d0d37280742bd", "f9e82e7dcf62447a925b7fb217686fab", "422b1b348e564e6ea223a8fff4f93b66", "44b0a31c7bf14d34a82470ec1b9510b5", "625146c091b14afeaec9a0fe5a61de0d", "f529b10febb845e6a7fbfe421e259974", "163623d0a1d24570a297203e8b5b3923", "d146b511de2e47289c91738fe1394f3e", "082768304fbc4a7ba9c1104025ff8cb5", "fdb41565d6cd496ca231e98779e1c823"]} id="-7EM78vxBh-H" outputId="939b7cb7-5554-4801-8668-8cf122f50719"
Map = geemap.Map(height="400pt")
Map
```

<!-- #region id="t8u-puRyBzzq" -->
## **Define a Region of Interest**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 223} id="R15YovkpB1W5" outputId="0602b150-adcf-4aed-ba1a-1b5f67829497"
# Read the shapefile of the West Bengal state using geopandas
shp_path = r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/West_Bengal_Boundary/District_shape_West_Bengal.shp"
wb_gdf = gpd.read_file(shp_path)
print(wb_gdf.shape)
wb_gdf.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 89} id="RqFgtywgB53Y" outputId="b0b339cb-ed31-463d-9df7-726a5e76b84e"
# Filter the 'Bankura' district from the geodataframe
roi_gdf = wb_gdf[wb_gdf['NAME']=="Bankura"]
roi_gdf
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="e6dCxSWUB8MZ" outputId="06e117a4-86d7-4a99-ebef-41ceee103483"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="TApngSNF7Uze" outputId="471a0e9e-002f-4adc-af7f-7681e9249bf4"
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

```python id="od1urzmGHKAz" colab={"base_uri": "https://localhost:8080/", "height": 17} outputId="b2365a0e-b0c4-4f6a-c7bb-48ba66d8830e"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="gxVsL_S98Tx-" outputId="1465d714-819e-4ba7-cd43-0c008275e639"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="bfTmcvEosbpI" outputId="2826f98d-09f6-4398-a8ce-da6252a7ff88"
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
### **Implementation on an Image Collection**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="NIPsgMeXHo49" outputId="ed23e391-5533-4e8c-81b4-2a5ba712fcf3"
# Apply 'renameL9', 'maskL9CloudsAndShadows', and 'calculateIndices' function on the whole image collection
# to create a cloud free median composite spectral indices image of the whole year
medianIndicesL9= L9Filtered.map(renameL9)\
                           .map(maskL9CloudsAndShadows)\
                           .map(calculateIndices)\
                           .median()\
                           .clip(roi_ee)

# Display only the NDVI image
NDVI_vis = {
    "min": 0,
    "max": 0.3,
    "bands": ["NDVI"],
    "palette": ["#a50026", "#da372a", "#f67b4a", "#fdbf6f", "#feeea2",
                "#eaf6a2", "#b7e075", "#74c365", "#229c52", "#006837"]
}

Map.addLayer(medianIndicesL9, NDVI_vis, "L9 Median NDVI")
```

<!-- #region id="bmuvhj3ZDRc7" -->
## **Download the Full Image to Drive**
<!-- #endregion -->

<!-- #region id="gglVrDz3FqN2" -->
🤔 **Note:** Prior to downloading the image, it is crucial to explicitly specify the projection of the image.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="u-yxJ4s2Fz13" outputId="f658cad4-20d5-411f-e214-4de69c6eeeb7"
# Define the projection
proj = ee.Projection("EPSG:32645")

# Reproject the 'medianIndicesL9' image
medianIndicesL9 = medianIndicesL9.reproject(proj, crsTransform=None, scale=30)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="kR6h42LeGRQB" outputId="1de40de8-fb36-4cee-ba16-8a310406e24b"
# Check the current projection and scale of the image
print("Projection Information:", medianIndicesL9.projection().getInfo())
print("Spatial Resolution:", medianIndicesL9.projection().nominalScale().getInfo())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 84, "referenced_widgets": ["78e6d5a3c19e4d0fa4f4d520e989a5ee", "12b5483d9716435d8fc75d0d5e471da0", "bfb05cf8a6ba4d2c9320210f07c3e5a6", "0de7a765594e489e94ddcd555b6af3f4", "f529871ce08b4ef6b6fcc9fe361194af", "2c854e9a053a48139b2f3a57152e04dc", "0f3303a4c4b3442baf571040ca286e7a", "9ba7c2dda5514e6ba2969e0566ff50d7", "5df0ac03709a4d36970f1eaf2f6e4e6b", "bb35f32ed61944e2b591ad4111002984", "ce85141d757740fab005ce3aeab9a659"]} id="hJtxYYkqEQ7M" outputId="b60532c1-fd32-4d6b-98f9-35ae520778b6"
# Specify the output file path and file name
output_path = "/content/drive/MyDrive/GEE//"
file_name = "Median_Indices_2022.tif"
geemap.download_ee_image(image=medianIndicesL9,
                         filename=output_path+file_name,
                         region=roi_ee.geometry(),
                         scale=30)
```

<!-- #region id="aTbKXcTRJVlQ" -->
## **Read the Image using Rasterio**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="MwPqqW9qJdpR" outputId="2993115a-7617-4794-d661-9c5759cdf199"
# Read the 'Median_Indices_2022.tif' image
src = rasterio.open(output_path+file_name)

# Store all the metadata
raster_meta = src.meta
driver = raster_meta["driver"]
dtype = raster_meta["dtype"]
nodata = raster_meta["nodata"]
width = raster_meta["width"]
height = raster_meta["height"]
count = raster_meta["count"]
crs = raster_meta["crs"]
transform = raster_meta["transform"]

# Print the metadata of the raster
src.meta
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="YmhCqEPaLtuo" outputId="846dfb2c-2cb7-40ff-faa1-28043f2d7f67"
# Store all the band names in a variable
band_names = src.descriptions
band_names
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="J4a42pArKf7K" outputId="baecd843-590d-4cc1-cc65-d3071fee2bfb"
# Convert the raster image into an array
raster_arr = src.read()

# Close the source file
src.close()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 452} id="d8wVTgmtKrA8" outputId="731f696a-e79d-4977-d72d-6e469bd70564"
# Plot the NDVI image
plt.figure()
plt.imshow(raster_arr[0], cmap="RdYlGn")
plt.colorbar(label="NDVI")
plt.title("Median NDVI 2022")
plt.show()
```

<!-- #region id="nXNeANOXQCZR" -->
🤔 **Note:** <br>When working with raster images, especially in scenarios involving larger regions of interest (ROIs) and high spatial resolution, it's imperative to optimize the processing strategy for efficiency. Processing the entire image at once can be computationally intensive, leading to increased memory usage and slower performance. To overcome these challenges, a more effective approach involves dividing the image into smaller tiles and processing each tile individually. This not only conserves memory but also significantly improves processing speed.
<!-- #endregion -->

<!-- #region id="rXcW3uHjQh8k" -->
## **Download the Image by Tiles**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="TYfMqk7MSlA6" outputId="45cbafc9-987f-4e19-bf61-8f38d9792722"
# Extract the bounding box of the ROI as a polygon
roi_boundary = roi_ee.geometry().bounds()

# Create a mesh grid
mesh = geemap.fishnet(roi_boundary, rows=2, cols=2, delta=0)
Map.addLayer(mesh.style(**vis_params), {}, "Mesh Grid")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 232, "referenced_widgets": ["b6f95862fb2b46ee9434f9683cea279e", "4abc7204e00747a494eb601d3c6775e0", "7191b33441c243709bbc4a424b8c4090", "7c5ac1bc7ad64a6097f5fad3bc321e98", "2b731bfcfdcd47d8839490d1fdb4d1cb", "3b8d39c9654241bdb27db0afb09155d1", "534c8f3a541543c3a9faf223f20646c6", "f11b2e63806f48e08807458470a6ea5a", "22d5f253b19a4d3c8a5f1238620edfa9", "2ba9c97265fa4250ad424349aea7ddcf", "4591fabbbb1140128538f9f03c375729", "7347c40cf2c749e68851d029ade7c553", "2173c4c33cc84b77b36ee47858aebfe1", "acf813a7d5c144e88679a84fe1d746ef", "cec96e00778d46fbb8d9c08305b18ce8", "0ca50f2d9eab4099a3215d46d03e89ca", "fd6c5d312825494482f46715f64fdebd", "2c7dccccebf146f891ade2b83059c1fa", "224293e188e44ed9b47652c2415adf4c", "44043d51897b44d8a73b39b589f0dc7b", "e14a832eee834959b875debe8f1611c5", "99892987244247fcb6913473995f7139", "b4e88e23c12e4256912ce07dd32eff2c", "cb441c5a492b47d4930df43fc62db27b", "30dcec8c51534a04a7a220003a8e5025", "297c032db6fc4c03bf4c52996f71dc91", "4947659d336d4d3ca20edea8c8cc6e4d", "7a0c8678f1264fdd8fc9d2d1176aa362", "46dd069111374266b1e5d2a58568b7b9", "31a5ac368301435f89bb15cf1ae23059", "b54f81d877d643719426a1503be72211", "30fdc48b97e748558893a4be307bb848", "3af11acd9c0b421ab4da83ffd83c894e", "d9c93ace142a4169981ebdfac06826e2", "30bc3a95026840788211be12f10d7bbd", "b9bf2e1fe0014a9a9824a40e6c47e208", "128e8609d7ae44f48bd201e32ac02db5", "d30a07aa7bed4e6182a665db4e19ff69", "767782f77b6947798c54213afa637b63", "883d03614e7647c48c29776d6e076bb3", "c5c6a92f959f48b092597b905a32e59a", "3468eff9945b4189bb6b7661dfed9045", "555732f35ba8416d896f228a164e9c40", "fe5dd8e6817e47c4adc233727a8c38d0"]} id="91I7l0WPUBsl" outputId="a21e65d0-72ff-40fb-de3a-89512c7a2e4d"
# Download the image by tiles
geemap.download_ee_image_tiles(image=medianIndicesL9,
                               features=mesh,
                               out_dir=output_path,
                               prefix="Median_Indices_2022_",
                               crs="EPSG:32645",
                               scale=30)
```

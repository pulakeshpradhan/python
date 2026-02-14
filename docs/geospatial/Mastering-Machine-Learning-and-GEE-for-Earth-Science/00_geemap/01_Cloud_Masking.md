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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/00_geemap/01_Cloud_Masking.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="6-7J5immvJK9" -->
# **Cloud Masking**
<!-- #endregion -->

<!-- #region id="RchKugASxtMI" -->
## **Import the Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="jtCpeK0V3Dcx" outputId="8cbc8784-d36c-4671-9659-048ed53c97cf"
from google.colab import drive
drive.mount("/content/drive")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="LSZlAfcbxddN" outputId="7c189744-8ded-4d83-9980-86ece48b6327"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="Cp8Jph4818M3" outputId="6b7eb743-2844-4d5b-82f3-c3abf22e6050"
# # Trigger the authentication flow.
# ee.Authenticate()

# # Initialize the library.
# ee.Initialize(project='my-project')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 554, "referenced_widgets": ["aa90900599f045fe89113f08603f8a58", "8d6ae0b4491a485d8a150a474b4f92ae", "d3628fd215d04c8794f1cd58e9ae7267", "9263df32698c4b0dbed985f150ee5b20", "237b11edd249453da0b90fa7d2a74b38", "1aca3668a65b4d718e7bc5f687cf29e0", "3b305815426c4729b9b2ddce25029b02", "1dc6e13a693f41e4afc63d30554573a9", "3a1d9a3f210f422c8253727dc2a1021b", "297c44e1e8c04e0fb77d5a96c04eac6e", "842ccaeac37a409faef4f45391ac1429", "cfcdc79bd79b4f1cb6039fcb8d7250e0", "f8722942c19e456098fb7bdca65aa6c7", "e265a0f26bca490380e04ebe885256fe", "e5faff476b6a4747bc3f6b220e7196b8", "0d0591a71fde45d69e06d3c838696015", "1e46d1659b1144ba9c7cd6dd631bf0a0", "cf7343bc47e948fab62a7d79c13454d7", "5395da550158409784dc679990a28b9f", "e45e192e43a0482586082222f908ebaa", "d114815022d54e0dbbec9babc82e027d", "04df554b78694b2dbe894674ff5f20eb", "3ce841333c6c402a90c3708b24b78f2f", "a0d8ed46ac5c40ac9ca9553595795cfc", "dbfa87997d3740d1a776512813764d2b", "9c825794fb544595a605bb7f06f00304", "135972269b9d49c78bcae58967dc988c", "40cb076249a046fdb89a7f64a5b4ea64", "07fe3cafe88f48cbbcfef09ba767aab3", "859497b0d67d48718aa1b00255fdd212", "c6aab3c27eb3462ba4b3e000b36f93b7", "8f07faee1e6d4b80a8157a1638b6240b", "c2168c3ee4e84437b6fba2f9ede1f695", "b5fa81f1523649ca80f76656a62242b1", "66cc480486e24dd1972d2bab75115b8e", "11bd23945493429199b2f78412ddb4cb", "13568fb06ac04092846ebd0c1d5826fb", "d9b8c4c6b7ce445485765072084f0759", "cafed16ceaf14aeb824c950540518830", "a1734ea24d074fd99ae14d43854fd65a", "9b50f9dfac3748ccb0171ed0cc64bb98", "64c1c7d3711b4d0f882406aa6ed08a46", "ea3ed3db5343430087ace7ccb3e32050", "126da6372d0542da95bed6e1f7590a1c", "0f6b42730f7a404bb782e3991ff10319", "2e30788c23d64312872a0cbe858e31ef", "45cbdda8afe14936a76e17d29d588245", "9368db9c62d64ce981fa885b7876b034", "11d387980201411894e474c824f0615c", "9b14009bb2a0475a9e51578d5323f6cd", "a68ea6bbce634c50878fa47157fec829", "76896c324d4a4e5896d6bc991a4b42a4", "a743ac63720049719d73800e56fc6f06", "7c39bae4f09f4df29b3f8caa321b5c50", "b5f16d99452a4258a3eba40269a8dafd", "a44753a5f6eb47f08125288c0289efb0", "19adda7a05e744f5bd7968a6f1311acb", "08ff364872a94283bba19d34d34b6d29", "cc21a68ffe6b4cf5ae0cbfe66ac15009", "fab1b6ec16d344fb821505e875cd3630", "46314ffb293041f7b4304046de0d02ce", "b9625843322c4a68a0b665258509d84e", "23119c034e234b5a98407f3f4f026ab2", "3080f2bf265946cf920ed6372e395600", "6ad1f747d3654cb3b8e246a500c00892", "5818a71f66bf46168c51fea6ce85e9f9", "dce54e7d628d43ce93687a3d33de0cef", "8e2e6da5894f46f28b12c3aa03ec46ac", "60b69a8409de4d5e905b5f3c59c751c0", "c047b5560d9b4db7b284446ea7814fc0", "d551642d6aaa4ff6b972ac72eb55f8ef", "9db69e8c2af3486b971e9462fe8dfff2", "063f8efa4b844f6da6d2ab8d11dd0259", "564ee9bde6c4497984c5021335f75fae", "5b0b46892bde41d8b31743fe610286d3", "b4c60a5807be46d48a43803e01270c9f", "76c2cda9b624416f8d8e16f91f9c2a8c", "23f48b670a934846ac7bb6c847853646", "2971b1eeef65488089223fb570374d75", "0eb55da115e44e55a16c286cd46cc200", "db50f4d17a1a4f329e89e5668c3a1998", "1b6fd2b44fb0457cbbef6c07c02df53a", "8b24397888ac4405b5b376a60943216c", "d4b99aad3d5f490ab9a6543ef09bf6b1", "635aac4de8c1470181f63127179a1884", "b8bc526123bb40018f886432389470fd", "4ccc49da5e9146b594b02466fc405af3", "bb97f00e24554351b796d69b436d80d2", "d03e5c387ee34baa9c8b8289729e3ccc", "5cca1afb82e54a958878e9355c629089", "77c04a0c7fc34950870f2c157478ccac", "fffeb66f08604a32aac33634dfd7063c", "873274bae91e406fb325ec78a94ffb0d", "25ce740cc4944b9e936cf3c9ac3c32a7", "2cb2b0775f964c0bb684bfb29bdd74e3", "12a8497c299e47a799f7169ec3198313", "cc1729de4beb456e957ddcb59952d445", "64499a7c74fd4aa2835a15358d9a8e09", "5dc5aadf7f944bd1873364209aeefd79", "3f97ce776c134d52a267233fedb3210f", "a61523d9a40b466abce56e64ca46ec78", "81ad9647dc234495a78f2613da3ed137", "49f23da0974a4b659d68e06d4eecba3a", "42d31cbaddf748d89770de3913e83954", "c8fd517e1b1c4510bfc316186245b056", "bc527cdac37448a8b458e99b984f7640", "67b0e0984d3f44d2aba67d9fa0845355", "44b5867f660841c09bd93d894c82e462", "0b8451c6f8e141c7831bd67610bd2de0", "0ac807d8d8474162a9552248dc99e1da", "aba40691e3cd40b0a7f4fe4dbbb80486", "279d8bcbbd294eacac6884b34b1ff10e", "e8138fd6b9774ac1920fa8967559e5e0", "7b92c5ed4354419dbcfae2e7cea7e256", "2afd338e349e4096bfb5f36cc1c6d68d", "d1883c1dd2e54bb1a196380c6ed19b8a", "6a669b617c704b13825c68e91708a327", "f4c048f77c924c428d92b820ec112869", "700c9599ae284ebe86c0232b5e8bd025", "3b9593cf10c34da7a80dbf5aec8e02a4", "35ad30a4b87a4a249d4078ca52148947", "5ebbe173dfe542fd951dbb5ab59ff93b", "86ba2ae0b20645b882a19e65d00c0002", "5bdcb1ba86c343c9aae1763410d079a6", "c36c1d0c911b4f3bb6124196337570c2", "126ec5cdb49640239e0aa74325708372", "bcf36f1845224f4e851ddcf2b0daebdc", "619b03505c8f483baf126fea430e2004", "f56d46cee4524658928cd66860181f1e", "f1bdb8c090e74996be40e642f682485b", "07bba0e9beb9455fa9643b8fc4089678", "43d5d72da19549da87e4d7ad18e92444", "a7af62ce23574138b2bca92a9fc9105a", "f28ec8e595244a6db5f04d2474fe0a8f", "8189e5c37f1743aa81bd8709fdd64496", "5a71c5c25489433189c9f563b14bd842", "cd7fc0baed544e66a8a016e6c75a5b7e", "6b525dc2d65b4485925426119302c8ad", "d7462d03109a49dcbf68a9d60325c04d", "d1735a3ca3cc45f1b945da17d6510941", "f11b7a6d47d84148b64adce82efb3827", "0a6a91cd89644cdc804cfad429a59326", "63edef84b67e4a758edaf269dd51723d", "5b747fc05c0742329976dcce16a11851", "4a247bc7e7304b78a2319b19f7bab323", "310f7edb967e47928b804734a12d390a", "ffb2d0746b7649c0bd2f661aa4b3eebd", "9ceaa73a1e614c60a36b35b9de5583d9", "3fe54d959f4c4d3dafb6f1100ba7d07b", "b03620417a9e4e699b3b4c6ee6a99cee", "186d522648634d5397f8c4bf38199565", "d49f315bb02e4e1cba962ce367afdaf7", "01af14b3946849519c98aa035c04319e", "057ebc2dbc01473694315e9863b96bda", "12e5f5fe96e64248bf1f10408462c2f8", "befc07c6f37449929449935b1bd0356f", "b31b7e46e1904ac187caad6a41873428", "b28adef0ed3f41768a28ffb42f3dc4ab", "130d5fa9352a42b2a7c0a7055032f667", "9e876da522114d72883d7f1c2c3b565f", "2a4d807d46704d0691eef591ec3c6afa"]} id="0txWpZFG1yz3" outputId="0e32fa34-1a19-4ca0-ce52-546bc3b6e6a2"
Map = geemap.Map(height="400pt")
Map
```

<!-- #region id="ncLmgxwd2gh9" -->
## **Define a Region of Interest**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 223} id="t4XvkoER3mx8" outputId="c7236a32-8c27-4868-81a2-db373428000b"
# Read the shapefile of the West Bengal state using geopandas
shp_path = r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/West_Bengal_Boundary/District_shape_West_Bengal.shp"
wb_gdf = gpd.read_file(shp_path)
print(wb_gdf.shape)
wb_gdf.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 89} id="py8GOFDG4Vcx" outputId="209eaf85-ccec-43bb-c74a-29e3160b74ad"
# Filter the 'Bankura' district from the geodataframe
roi_gdf = wb_gdf[wb_gdf['NAME']=="Bankura"]
roi_gdf
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="q1bvvzFT4tt1" outputId="35753df8-8c69-459a-91ed-2597e27aa851"
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
## **Cloud Masking on Landsat Data**
<!-- #endregion -->

<!-- #region id="KMvvsJdxHTnz" -->
### **Filtering Landsat 9 Image Collection**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="TApngSNF7Uze" outputId="0d268ac7-6d97-43cc-c513-508cb3741f65"
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
### **Function to Remove Clouds and Shadows**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="gxVsL_S98Tx-" outputId="3cc9abfc-38c9-415f-99b1-d6c63b338637"
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

<!-- #region id="mxogLTKiIQzi" -->
🤔 **Note:** <br>
1. **`bitwiseAnd` Function:**<br>
The `bitwiseAnd` function is used to perform a bitwise `AND` operation on the bits of two numbers or image bands. It takes two operands and returns a result where each bit position in the output is the logical AND of the corresponding bits in the input operands.

2. **`eq` Function:**<br>
The `eq` function is used for element-wise equality comparison. It returns a binary image where each pixel is set to 1 if the corresponding pixels in the input images are equal and 0 otherwise.
<!-- #endregion -->

<!-- #region id="iforZ9iNHfvE" -->
### **Implementation on an Image and Image Collection**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="ipP7tUXVFrQJ" outputId="cd894d10-6665-4ae8-e3a5-0fa5a3de3080"
# Check the cloud cover value of the first image
# L9Filtered.first()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="mmYyHq2jF8Dq" outputId="30f65580-e557-4e9e-f258-93a2b19ad5d6"
# Display the first image of the filtered image collection
L9FilteredFirst = L9Filtered.first()

# Display a Standard False Color Composite (SFCC)
sfcc_vis = {
    "min": 8000,
    "max": 17000,
    "bands": ["SR_B5", "SR_B4", "SR_B3"]
}
Map.addLayer(L9FilteredFirst, sfcc_vis, "Landsat 9 Image")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="qiREfQOiEfRw" outputId="2e58c8cb-49fc-4204-9314-ac866ca78058"
# Apply the 'maskL9CloudsAndShadows' function on the first image of the filtered image collection
cloud_free_image = maskL9CloudsAndShadows(L9FilteredFirst)
Map.addLayer(cloud_free_image, sfcc_vis, "Landsat 9 Cloud Free Image")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="NIPsgMeXHo49" outputId="2c4d2268-692c-41b4-cecc-1765b2f783d5"
# Apply the 'maskL9CloudsAndShadows' function on the filtered image collection
# to create a cloud free composite image of the whole year
cloud_free_composite = L9Filtered.map(maskL9CloudsAndShadows)\
                                 .median()\
                                 .clip(roi_ee)
Map.addLayer(cloud_free_composite, sfcc_vis, "Landsat 9 Cloud Free Composite")
```

<!-- #region id="4Ahq7TjlKesy" -->
🤔 **Note:** The `map` function in GEE is commonly used for applying a specified function to each element of a collection. This function is particularly useful for processing each image in an image collection or each feature in a feature collection.
<!-- #endregion -->

<!-- #region id="tsGYYNzE9Tum" -->
## **Cloud Masking on Sentinel Data**
<!-- #endregion -->

<!-- #region id="4gjFQZcg95NC" -->
### **Filtering Sentinel 2 Image Collection**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="5l0meMYm9arn" outputId="7024ae3d-f16b-42f4-85a5-cbc8bdf1f6ca"
# Read Landsat 9 image collection from EE
S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

# Filter the image collection with roi, daterange, and cloud cover property
S2Filtered = S2.filterBounds(roi_ee)\
               .filterDate("2022-01-01", "2022-12-31")\
               .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 50)

# Print the size of the filtered image collection
S2Filtered.size().getInfo()
```

<!-- #region id="NRE1iYbP-CKq" -->
### **Function to Remove Clouds and Shadows**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="Z1CO_S5R-LDN" outputId="0c7e3367-9605-4eef-aae2-54b7c2cc0bc8"
# Write a function to remove clouds from Sentinel 2 imagery
def maskS2CloudsAndShadows(image):

    # Select the 'MSK_CLDPRB', 'MSK_SNWPRB', and 'SCL' bands
    cloudProb = image.select("MSK_CLDPRB")
    snowProb = image.select("MSK_SNWPRB")
    scl = image.select("SCL")

    # Define the thresholds for cloud an snow probability
    cloudMask = cloudProb.lt(10)
    snowMask = snowProb.lt(10)

    # Mask the 'cloud shadow' and 'cirrus' pixels from 'SCL' band
    cloudShadowMask = scl.eq(3) # Cloud Shadow = 3
    cirrusMask = scl.eq(10) # Cirrus = 10

    # Define the final mask
    mask = cloudMask.And(snowMask)\
                    .And(cloudShadowMask.neq(1))\
                    .And(cirrusMask.neq(1))

    return image.updateMask(mask)
```

<!-- #region id="-yHM8X_BCD0P" -->
### **Implementation on an Image and Image Collection**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="bheQhqIXCGYu" outputId="38c07264-65b7-46bf-a2b3-21fbba1750ca"
# Select an image from the filtered image collection of the monsoon time
# where 'CLOUDY_PIXEL_PERCENTAGE' value is in between 40 to 50
S2_cloud_image = S2Filtered.filterBounds(roi_ee)\
                        .filterDate("2022-06-01", "2022-08-31")\
                        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "greater_than", 40)\
                        .first()

# Display a Standard False Color Composite (SFCC) of the cloud image
S2_sfcc_vis = {
    "min": 0,
    "max": 3000,
    "bands": ["B8", "B4", "B3"]
}
Map.addLayer(S2_cloud_image, S2_sfcc_vis, "Sentinel 2 Cloud Image")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="Ft0PMTHzHih2" outputId="c129513e-6e6c-4584-a994-5d270a4520b6"
# Apply the 'maskS2CloudsAndShadows' function on the cloud image
S2_cloud_free_image = maskS2CloudsAndShadows(S2_cloud_image)
Map.addLayer(S2_cloud_free_image, S2_sfcc_vis, "Sentinel 2 Cloud Free Image")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="yyV1AtvlIP-K" outputId="8628d05d-c097-4c64-8dc7-dd250d4a3c9e"
# Apply the 'maskS2CloudsAndShadows' function on the filtered image collection
# to create a cloud free composite image of the whole year
S2_cloud_free_composite = S2Filtered.map(maskS2CloudsAndShadows)\
                                    .median()\
                                    .clip(roi_ee)
Map.addLayer(S2_cloud_free_composite, S2_sfcc_vis, "Sentinel 2 Cloud Free Composite")
```

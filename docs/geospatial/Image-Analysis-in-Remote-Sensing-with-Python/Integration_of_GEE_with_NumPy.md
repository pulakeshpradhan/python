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
<a href="https://colab.research.google.com/github/geonextgis/Image-Analysis-in-Remote-Sensing-with-Python/blob/main/Integration_of_GEE_with_NumPy.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="Tt8UZsrEPptG" -->
## **Importing Required Libraries**
<!-- #endregion -->

```python id="IbHHSuYwLNN5"
import ee
import geemap
import matplotlib.pyplot as plt
import seaborn as sns
```

```python id="fUanfUHROhbM"
# Initializing Earth Engine
ee.Initialize()
```

<!-- #region id="-OasPhgXQW_n" -->
## **Defining the Region of Interest**


<!-- #endregion -->

```python id="4CtS3B7bOkgN"
roi = ee.FeatureCollection("users/geonextgis/Bankura_District")
```

```python id="q1AtxYpPPD-w"
# Initializing a map
map = geemap.Map(height="450px")
map.addLayer(roi)
map.centerObject(roi, 9)
```

<!-- #region id="mxCjG-K6Qmh0" -->
## **Preparing the Dataset**
<!-- #endregion -->

```python id="A0as1uk6Qi3D"
# Importing Landsat 9 image collection
l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

# Filtering the Landsat 9 image collection
filteredL9 = l9.filterBounds(roi)\
               .filterDate("2023-01-01", "2023-12-31")\
               .sort("CLOUD_COVER")\
               .first()\
               .select(["SR_B6", "SR_B5", "SR_B4"])\
               .multiply(0.0000275).add(-0.2)\
               .clip(roi)

# Creating a FCC visulalization
fccVis = {
    "min": 0.0,
    "max": 0.3,
    "bands": ["SR_B6", "SR_B5", "SR_B4"]
}
map.addLayer(filteredL9, fccVis, "Landsat 9 Image")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 471, "referenced_widgets": ["42d20527513b4e03b3b20d9690d3283a", "8d9cfe69e0d6461c9156852077dfc1ed", "8ec9cd2c5b70478698aac0fe23ebd57a", "d49c6eb44cc34b7fb4e6211e863ec924", "864c903719f141ee8bdad82d25bcc41e", "95b3f6ca3ef44b239966e8104581c47e", "bccac00a96ca467bbddd2cbad21569f2", "44ae83655de341dc87424168f58347cf", "33eca6f3f4de453e9920f1e77ed0e8be", "62584e5182ab4673a3a952ac8fe298a0", "5786358660214d3bb206f0d23fac400a", "c18015fabdbd4f7ab89cd534d6aaa83f", "59cb89040d7b4a7393f4b7b824223bae", "eb7c3573f971467ab178dcc9620db23d", "31dce48c538c4befaa6d59fdf5a6f802", "e4c3fa5003574bf696f92ef0c017622b", "66bd6d94a7704692a80aaae8443f478f", "99f81972f8c44840975910ff435b6288", "cc6314a4ae974effa27623398dcfa1d9", "0571fb7435cf4f58b85aa26bd6575d19", "4491a309d4134180be09952fc9705e90", "9d90f3b2616643f6920e1a0f5a078431", "d7de340d8e5d4a3faaa8e003d06215a8", "65f8eb0063e544a087fac7ac725374fa", "d867e36d41774fc587051e5ea04ad2db", "9b90b8eca25241c68025ba7961d6bb1e", "311cc9f60c01475a95cb52fad27d15ba", "2bea5869901e4db9b4622341557f2474", "732d938c98d4484c968bbb0b6826a91f"]} id="MJ43TOqTS_6l" outputId="ecee40c8-d60f-4166-fbc9-bd31178f6d5a"
# Extracting a subset area
roi_subset = ee.Geometry.Polygon([
                                 [87.011035, 23.179817],
                                 [87.104422, 23.179817],
                                 [87.104422, 23.266581],
                                 [87.011035, 23.266581],
                                 [87.011035, 23.179817]
                                 ])

# Clipping the image with the subset area
subset_image = filteredL9.clip(roi_subset)

# Add the subset image to the map
map.addLayer(subset_image, fccVis, "Subset Image")
map.addLayer(roi_subset, {"color": "red"}, "ROI Subset")

# Display the map
map
```

<!-- #region id="nmzo5doxwf2t" -->
## **Converting EE Image to NumPy Array**
<!-- #endregion -->

```python id="zoEFPVH8qjn9"
# Converting subset image to numpy array
image_arr = geemap.ee_to_numpy(subset_image)
```

```python colab={"base_uri": "https://localhost:8080/"} id="OrDx_13Qw2Va" outputId="f2d81652-f574-4fad-9a49-46676479a9bc"
# Checking the image array information
print(f"Datatype: {type(image_arr)}")
print(f"Shape: {image_arr.shape}")
print(f"Dimensions: {image_arr.ndim}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="DT_p6Pqmw_tD" outputId="a003b683-2d5b-4d3b-c7cb-85fd1b074e1b"
# Checking the numpy array
image_arr
```

```python colab={"base_uri": "https://localhost:8080/"} id="lqW460zk0gcs" outputId="4008f99b-15ee-4ed6-8bdb-f2ae86ed20ff"
# Getting the maximum and minimum pixel value in the image
print("Maximum Pixel Value in the image:", scaled_arr.max())
print("Minimum Pixel Value in the image:", scaled_arr.min())
```

```python colab={"base_uri": "https://localhost:8080/"} id="q6eWDxxlzaM-" outputId="7f0f3c1c-537c-4823-990a-f0bb53fe2d26"
# Scaling the value in the range of 0 to 255
scaled_arr = (scaled_arr[:, :, 0:3]/scaled_arr.max())*255

# Printing the datatype of the scaled array
type(scaled_arr.dtype)

# Changing the datatype to unsigned int8
scaled_arr = scaled_arr.astype("uint8")

# Printing the scaled array
scaled_arr
```

<!-- #region id="HH8lfwaa2_PM" -->
## **Data Visualization**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 291} id="S6ZGQTT23qXz" outputId="1e8c7680-eefb-4939-9a86-5e91574ec3e6"
# Writing a function to generate histogram of all bands in the array
def plot_histogram(array, bandNames):
  nBands = array.ndim
  fig, ax = plt.subplots(nrows=1, ncols=nBands, figsize=(14, 3))
  fig.subplots_adjust(wspace = 0.28)

  for i in range(nBands):
    band = array[:, :, i].ravel()
    sns.histplot(band, bins=50, ax=ax[i])
    ax[i].set_title(f"{bandNames[i]} Band's Histogram")
    ax[i].set_xlabel("Pixel Values")
    ax[i].title.set_size(10)

# Defining Band Names in a list
bandNames = ["SWIR1", "NIR", "Red"]

# Plotting the histogram of all the bands
plot_histogram(scaled_arr, bandNames)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="8L8r7C282780" outputId="59c346fa-5c89-47ae-8a6a-14ab26142192"
# Writing a function to generate image display of all bands in the array
def plot_image(array, bandNames):
  nBands = array.ndim
  fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

  for i in range(nBands):
    band = array[:, :, i]
    axs[i].imshow(band)
    axs[i].set_title(bandNames[i] + " Band")
    axs[i].title.set_size(10)

# Plotting the image of all the bands
plot_image(scaled_arr, bandNames)
```

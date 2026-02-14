[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/Flood-Susceptibility-Zonation-of-Malda-District/Notebooks/9_Convert_the_Image_into_CSV.ipynb)

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

<!-- #region id="YlguyHU_VXb_" -->
# **Convert the Image into CSV**
<!-- #endregion -->

<!-- #region id="OT4F-99YIVrp" -->
## **Import Required Libraries**
<!-- #endregion -->

```python id="mOG7H7d9KZxO"
# !pip install rasterio
```

```python id="c1KFEd57Hkbc"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import pickle
import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="P0OjWhV8ItSj" -->
## **Prepare the Image for Classification**
<!-- #endregion -->

```python id="9Z85JbvsIeOR"
# Read the image with Rasterio
image = rasterio.open("/content/drive/MyDrive/ML & DL/Flood Data/Maldah_Flood_Parameters.tif")
```

```python id="Yxxng30-I2GH"
# Store the image parameters in separate variables
bandNum = image.count
height = image.height
width = image.width
crs = image.crs
transform = image.transform
shape = (height, width)
```

```python colab={"base_uri": "https://localhost:8080/"} id="BOeJwyzST-0N" outputId="cefae966-ef54-4f18-935c-228174ef6987"
print("Band Number:", bandNum)
print("Image Height:", height)
print("Image Width:", width)
print("CRS:", crs)
print("Transform:\n", transform)
print("Shape:", shape)
```

```python id="dl8W7IFpI6-L"
# Create an empty pandas dataframe to store the pixel values
image_bands = pd.DataFrame()
```

```python id="jnEZHEaXI-mg"
# Joining the pixel values of different bands into the dataframe
for i in image.indexes:
    temp = image.read(i)
    temp = pd.DataFrame(data=np.array(temp).flatten(), columns=[i])
    image_bands = temp.join(image_bands)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="s9f6eEtJJK6G" outputId="8a4ccc2d-ddc5-4500-e2e1-89d9af71006c"
image_bands
```

```python id="ArD9T7RuKl6o"
# Store all the band names in a list
bandNames = ["Elevation", "Slope", "Dist_to_River", "Drainage_Density",
             "Geomorphology", "Lithology", "Relief_Amplitude", "Rainfall",
             "MFI", "NDVI", "MNDWI", "SPI", "STI", "TPI", "TRI", "TWI",
             "LULC", "Clay_Content"];
```

```python colab={"base_uri": "https://localhost:8080/", "height": 226} id="cSHWAzKpKyKI" outputId="49c2bf32-8fd3-4262-ae3c-a33bba8e0cba"
# Change the column names
image_bands.columns = bandNames[::-1]
image_bands.head()
```

<!-- #region id="NloXgCQOLhT6" -->
## **Data Preprocessing**
<!-- #endregion -->

```python id="eVPJ6oqRLfyh"
# Fill the null values of the Clay_Content column with 0
image_bands.fillna(0, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 443} id="T9sSI3iiM4sA" outputId="0ff7670b-612a-4c8e-9eac-cf9576559c77"
image_bands
```

```python id="WRv5SDqiM-jY"
# Rename the values of the categorical variables
# Define the values for the geomorphology
geomorpholoy_dict = {1: "Active_Flood_Plain",
                     2: "Embankment",
                     3: "Older_Alluvial_Plain",
                     4: "Older_Flood_Plain",
                     5: "Pond",
                     6: "River",
                     7: "WatBod_Lake",
                     8: "Younger_Alluvial_Plain"}

# Define the values for the lithology
lithology_dict = {1: "Cl_wi_S_Si_Ir_N",
                  2: "Fe_Ox_S_Si_Cl",
                  3: "S_Si_Gr",
                  4: "S_Si_Cl",
                  5: "S_Si_Cl_wi_Cal_Co"}

# Define the values for the LULC
lulc_dict = {1: "Waterbodies",
             2: "Natural_Vegetation",
             3: "Agricultural_Field",
             4: "Bare_Ground",
             5: "Built_UP_Area"}
```

```python id="kpXw_cgmNNzj"
image_bands.replace({"Geomorphology": geomorpholoy_dict, "Lithology": lithology_dict, "LULC": lulc_dict},
                     inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="E8T2UnEYOAsz" outputId="2eca1317-38d4-4527-cb5a-4c602a70db38"
# Apply OHE on 'Geomorphology', 'Lithology' and 'LULC' Columns*
image_bands = pd.get_dummies(image_bands, columns=["Geomorphology", "Lithology", "LULC"])
image_bands.info()
```

<!-- #region id="UuIjndx3OnKL" -->
## **Select the Best Features**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="41RNwp4LVxE3" outputId="91c8bf8f-21c5-412d-8182-08ede3ffcb75"
# Define the best features in a list
selected_features = ['Dist_to_River', 'TWI', 'Rainfall', 'Clay_Content', 'TRI', 'NDVI',
                     'MFI', 'Elevation', 'MNDWI', 'Drainage_Density',
                     'Geomorphology_Active_Flood_Plain',
                     'Geomorphology_Older_Alluvial_Plain', 'Geomorphology_Older_Flood_Plain',
                     'Lithology_Cl_wi_S_Si_Ir_N', 'Lithology_Fe_Ox_S_Si_Cl',
                     'Lithology_S_Si_Cl', 'Lithology_S_Si_Cl_wi_Cal_Co',
                     'LULC_Agricultural_Field', 'LULC_Built_UP_Area',
                     'LULC_Natural_Vegetation']
len(selected_features)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 443} id="49cHcn7jWA-B" outputId="96062b8b-87af-42b6-8fc5-6f562f3a09c8"
image_bands = image_bands[selected_features]
image_bands
```

<!-- #region id="h3XnSg-PXRhg" -->
## **Export the Data as CSV**
<!-- #endregion -->

```python id="D1bGi7UsXL6B"
output_folder = "/content/drive/MyDrive/ML & DL/"
file_name = "Image_CSV.csv"
```

```python id="qR2z-BYRXnBD"
# image_bands.to_csv(output_folder+file_name)
```

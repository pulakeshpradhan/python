[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/Flood-Susceptibility-Zonation-of-Malda-District/Notebooks/4_Training_and_Testing_Data_Preparation.ipynb)

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

# **Training and Testing Data Preparation**


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

## **Read the Data**

```python
# Read the flood samples data
gdf = gpd.read_file("D:\Research Works\Flood\Flood_Risk_Zonation_of_Maldah\Datasets\Shapefiles\Flood_Sample_Data.shp")
print(gdf.shape)
gdf.head()
```

## **Preprocess the Data**

```python
# Check the column informations
gdf.info()
```

```python
# Change the name of the columns
gdf.columns
```

```python
# Change the name of the columns
new_col_names = ['Relief_Amplitude', 'Dist_to_River', 'LULC', 'TWI', 'Rainfall', 
                 'Clay_Content', 'STI', 'TRI', 'TPI',
                 'SPI', 'NDVI', 'Slope', 'MFI', 'Elevation', 'Flood', 'MNDWI',
                 'Drainage_Density', 'Lithology', 'Geomorphology', 'geometry']

# Create a dictionary
new_col_dict = dict(zip(gdf.columns, new_col_names))

# Change the name of the columns
gdf.rename(columns=new_col_dict, inplace=True)
```

```python
gdf.head()
```

## **Rename the Values of the Categorical Variables**

```python
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

```python
gdf.replace({"Geomorphology": geomorpholoy_dict, "Lithology": lithology_dict, "LULC": lulc_dict},
             inplace=True)
gdf.head()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test, y_train, y_test = train_test_split(gdf.drop("Flood", axis=1),
                                                    gdf["Flood"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

## **Apply OHE on 'Geomorphology', 'Lithology' and 'LULC' Columns**

```python
# Apply One Hot Encoding on the training data
X_train_encoded = pd.get_dummies(X_train, columns=["Geomorphology", "Lithology", "LULC"])
X_train_encoded.info()
```

```python
# Apply One Hot Encoding on the testing data
X_test_encoded = pd.get_dummies(X_test, columns=["Geomorphology", "Lithology", "LULC"])
X_test_encoded.info()
```

## **Reset the Index of the Dataframe and Series**

```python
X_train_encoded.reset_index(drop=True, inplace=True)
X_test_encoded.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
```

## **Finalize the Training and Testing Data**

```python
# Add the y_train to X_train_encoded
X_train_encoded["Flood"] = y_train
training_df = X_train_encoded
```

```python
training_df.info()
```

```python
# Add the y_test to X_test_encoded
X_test_encoded["Flood"] = y_test
testing_df = X_test_encoded
```

```python
testing_df.info()
```

```python
training_df.head()
```

```python
testing_df.head()
```

## **Export the Training and Testing Data**

```python
output_folder_csv = "D:\\Research Works\\Flood\\Flood_Risk_Zonation_of_Maldah\\Datasets\\CSVs\\"
output_folder_shp = "D:\\Research Works\\Flood\\Flood_Risk_Zonation_of_Maldah\\Datasets\\Shapefiles\\"
```

```python
# Export as CSV files
# training_df.to_csv(output_folder_csv+"Training_Data.csv")
# testing_df.to_csv(output_folder_csv+"Testing_Data.csv")
```

```python
# Export as SHP files
training_gdf = gpd.GeoDataFrame(training_df, geometry=training_df.geometry)
testing_gdf = gpd.GeoDataFrame(testing_df, geometry=testing_df.geometry)

# training_gdf.to_file(output_folder_shp+"Training_Data.shp", driver="ESRI Shapefile")
# testing_gdf.to_file(output_folder_shp+"Testing_Data.shp", driver="ESRI Shapefile")
```

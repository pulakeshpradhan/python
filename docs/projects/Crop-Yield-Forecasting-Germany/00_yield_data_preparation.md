[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/Crop-Yield-Forecasting-Germany/00_yield_data_preparation.ipynb)

# **DE Yield Data Preparation**

## **Import Dependencies**


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from tqdm.auto import tqdm
from difflib import SequenceMatcher
import os
import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
```

## **Read the Datasets**


```python
# Read the shapefile for DE NUTS3
de_nuts_gdf = gpd.read_file(r"D:\GITHUB\crop-yield-prediction-germany\datasets\shapefiles\DE_NUTS\DE_NUTS_3.shp")
print(de_nuts_gdf.shape)
de_nuts_gdf.head()
```

    (455, 9)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NUTS_ID</th>
      <th>LEVL_CODE</th>
      <th>CNTR_CODE</th>
      <th>NAME_LATN</th>
      <th>NUTS_NAME</th>
      <th>MOUNT_TYPE</th>
      <th>URBN_TYPE</th>
      <th>COAST_TYPE</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DE11B</td>
      <td>3</td>
      <td>DE</td>
      <td>Main-Tauber-Kreis</td>
      <td>Main-Tauber-Kreis</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>POLYGON ((1074230.536 6408356.046, 1073820.827...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DE11C</td>
      <td>3</td>
      <td>DE</td>
      <td>Heidenheim</td>
      <td>Heidenheim</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>MULTIPOLYGON (((1131091.261 6235073.568, 11312...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DE11D</td>
      <td>3</td>
      <td>DE</td>
      <td>Ostalbkreis</td>
      <td>Ostalbkreis</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>MULTIPOLYGON (((1141777.678 6284962.486, 11412...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DE121</td>
      <td>3</td>
      <td>DE</td>
      <td>Baden-Baden, Stadtkreis</td>
      <td>Baden-Baden, Stadtkreis</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>MULTIPOLYGON (((910859.613 6248068.047, 913127...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DE122</td>
      <td>3</td>
      <td>DE</td>
      <td>Karlsruhe, Stadtkreis</td>
      <td>Karlsruhe, Stadtkreis</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>POLYGON ((938225.711 6286986.826, 940668.057 6...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Read the latest yield data for Germany for the year 2022, and 2023
de_yield_2023 = pd.read_excel(
       r"D:\GITHUB\crop-yield-prediction-germany\datasets\csvs\DE_Yield_Latest.xlsx", 
       skiprows=[i for i in range(7)], nrows=537,
       header=None,
       names=["district_no", "district", "ww", "rye",
              "wb", "sb", "oats", "triticale",
              "pota_tot", "sugarbeet", "wrape", "silage_maize"])

de_yield_2023.replace(["-", "", "/", ".", "..."], np.nan, inplace=True) # replace the special characters
de_yield_2023["district_no"] = de_yield_2023["district_no"].astype("int") # change the datatype of district no into 'int'
de_yield_2023 = pd.melt(
       de_yield_2023,
       id_vars='district_no', 
       value_vars=de_yield_2023.columns[2:], 
       var_name='var', ignore_index=True
) # melt the dataframe
de_yield_2023['year'] = 2023 # add the 'year' info
de_yield_2023['measure'] = 'yield' # add the 'measure' column
de_yield_2023['outlier'] = np.nan # add the outlier columns
print(de_yield_2023.shape)
de_yield_2023.head()
```

    (5370, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district_no</th>
      <th>var</th>
      <th>value</th>
      <th>year</th>
      <th>measure</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>ww</td>
      <td>83.2</td>
      <td>2023</td>
      <td>yield</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>ww</td>
      <td>NaN</td>
      <td>2023</td>
      <td>yield</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002</td>
      <td>ww</td>
      <td>NaN</td>
      <td>2023</td>
      <td>yield</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1003</td>
      <td>ww</td>
      <td>85.2</td>
      <td>2023</td>
      <td>yield</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>ww</td>
      <td>NaN</td>
      <td>2023</td>
      <td>yield</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
de_yield_2022 = pd.read_excel(
       r"D:\Research Works\Agriculture\Germany_Multiple_Crops_Cimate_Change\Datasets\Yield_Statistitics_WholeGermany_1999-2022\Yield_Statistitics_WholeGermany_2022.xlsx", 
       skiprows=[i for i in range(7)], nrows=537,
       header=None,
       names=["district_no", "district", "ww", "rye",
              "wb", "sb", "oats", "triticale",
              "pota_tot", "sugarbeet", "wrape", "silage_maize"])

de_yield_2022.replace(["-", "", "/", ".", "..."], np.nan, inplace=True) # replace the special characters
de_yield_2022["district_no"] = de_yield_2022["district_no"].astype("int") # change the datatype of district no into 'int'
de_yield_2022 = pd.melt(
       de_yield_2022,
       id_vars='district_no', 
       value_vars=de_yield_2022.columns[2:], 
       var_name='var', ignore_index=True
) # melt the dataframe
de_yield_2022['year'] = 2022 # add the 'year' info
de_yield_2022['measure'] = 'yield' # add the 'measure' column
de_yield_2022['outlier'] = np.nan # add the outlier columns
print(de_yield_2022.shape)
de_yield_2022.head()
```

    (5370, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district_no</th>
      <th>var</th>
      <th>value</th>
      <th>year</th>
      <th>measure</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>ww</td>
      <td>95.8</td>
      <td>2022</td>
      <td>yield</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>ww</td>
      <td>NaN</td>
      <td>2022</td>
      <td>yield</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002</td>
      <td>ww</td>
      <td>NaN</td>
      <td>2022</td>
      <td>yield</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1003</td>
      <td>ww</td>
      <td>101.7</td>
      <td>2022</td>
      <td>yield</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>ww</td>
      <td>NaN</td>
      <td>2022</td>
      <td>yield</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Read the yield dataset available for DE from 1979 to 2021
de_yield_1979_21 = pd.read_csv(r"D:\GITHUB\crop-yield-prediction-germany\datasets\csvs\openagrar_derivate_00056476\Final_data.csv")

# Filter the dataframe for yield only
de_yield_1979_21 = de_yield_1979_21[de_yield_1979_21['measure']=='yield']

# Create a seperate df to store only 'district_no', 'district', and 'nuts_id'
de_nuts = de_yield_1979_21[['district_no', 'district', 'nuts_id']].drop_duplicates()

print(de_yield_1979_21.shape)
de_yield_1979_21.head()
```

    (179691, 8)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district_no</th>
      <th>district</th>
      <th>nuts_id</th>
      <th>year</th>
      <th>var</th>
      <th>measure</th>
      <th>value</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>grain_maize</td>
      <td>yield</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>oats</td>
      <td>yield</td>
      <td>4.95</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>potat_tot</td>
      <td>yield</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>rye</td>
      <td>yield</td>
      <td>4.35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>sb</td>
      <td>yield</td>
      <td>3.63</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## **Data Processsing**


```python
# Merge the district information to the latest dataset
de_yield_2023 = pd.merge(left=de_yield_2023, right=de_nuts, on='district_no', how='inner')
de_yield_2022 = pd.merge(left=de_yield_2022, right=de_nuts, on='district_no', how='inner')

# Convert the yield values from dt/ha to t/ha
de_yield_2023['value'] = de_yield_2023['value'] / 10
de_yield_2022['value'] = de_yield_2022['value'] / 10

# Reorder the columns based on the data from 1971-2021
de_yield_2023 = de_yield_2023[de_yield_1979_21.columns]
de_yield_2022 = de_yield_2022[de_yield_1979_21.columns]

print('Shape of the data for 2023:', de_yield_2023.shape)
print('Shape of the data for 2022:', de_yield_2022.shape)
```

    Shape of the data for 2023: (3970, 8)
    Shape of the data for 2022: (3970, 8)
    

## **Merge All the Datasets**


```python
# Concat all the datasets
merged_df = pd.concat((de_yield_1979_21, de_yield_2022, de_yield_2023), ignore_index=True)

# Sort the dataframe based on district number, year, and var
merged_df.sort_values(by=['district_no', 'year', 'var'], inplace=True)
print(merged_df.shape)
merged_df.head()
```

    (187631, 8)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district_no</th>
      <th>district</th>
      <th>nuts_id</th>
      <th>year</th>
      <th>var</th>
      <th>measure</th>
      <th>value</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>grain_maize</td>
      <td>yield</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>oats</td>
      <td>yield</td>
      <td>4.95</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>potat_tot</td>
      <td>yield</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>rye</td>
      <td>yield</td>
      <td>4.35</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>sb</td>
      <td>yield</td>
      <td>3.63</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_df[(merged_df['year']==2022) & (merged_df['var']=='ww')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district_no</th>
      <th>district</th>
      <th>nuts_id</th>
      <th>year</th>
      <th>var</th>
      <th>measure</th>
      <th>value</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179691</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>2022</td>
      <td>ww</td>
      <td>yield</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>179692</th>
      <td>1002</td>
      <td>Kiel, kreisfreie Stadt</td>
      <td>DEF02</td>
      <td>2022</td>
      <td>ww</td>
      <td>yield</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>179693</th>
      <td>1003</td>
      <td>Lübeck, kreisfreie Stadt</td>
      <td>DEF03</td>
      <td>2022</td>
      <td>ww</td>
      <td>yield</td>
      <td>10.17</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>179694</th>
      <td>1004</td>
      <td>Neumünster, kreisfreie Stadt</td>
      <td>DEF04</td>
      <td>2022</td>
      <td>ww</td>
      <td>yield</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>179695</th>
      <td>1051</td>
      <td>Dithmarschen, Landkreis</td>
      <td>DEF05</td>
      <td>2022</td>
      <td>ww</td>
      <td>yield</td>
      <td>9.47</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>180083</th>
      <td>16073</td>
      <td>Saalfeld-Rudolstadt, Landkreis</td>
      <td>DEG0I</td>
      <td>2022</td>
      <td>ww</td>
      <td>yield</td>
      <td>4.86</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>180084</th>
      <td>16074</td>
      <td>Saale-Holzland-Kreis</td>
      <td>DEG0J</td>
      <td>2022</td>
      <td>ww</td>
      <td>yield</td>
      <td>6.16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>180085</th>
      <td>16075</td>
      <td>Saale-Orla-Kreis</td>
      <td>DEG0K</td>
      <td>2022</td>
      <td>ww</td>
      <td>yield</td>
      <td>5.69</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>180086</th>
      <td>16076</td>
      <td>Greiz, Landkreis</td>
      <td>DEG0L</td>
      <td>2022</td>
      <td>ww</td>
      <td>yield</td>
      <td>6.64</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>180087</th>
      <td>16077</td>
      <td>Altenburger Land, Landkreis</td>
      <td>DEG0M</td>
      <td>2022</td>
      <td>ww</td>
      <td>yield</td>
      <td>8.38</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>397 rows × 8 columns</p>
</div>



## **Plot the Dataset in Maps**


```python
def plot_var_map(dataframe, shapefile, var, year):

    # prepare the data 
    dataframe = dataframe[(dataframe['year']==year) & (dataframe['var']==var)]
    dataframe = dataframe[['nuts_id', 'district_no', 'district', 'year', 'var', 'value']]

    shapefile.rename(columns={'NUTS_ID': 'nuts_id'}, inplace=True)

    # merged the dataframe with the shapefile
    shapefile_merged = pd.merge(left=shapefile, right=dataframe, on='nuts_id', how='left')
    shapefile_merged = shapefile_merged[['nuts_id', 'district_no', 'NUTS_NAME', 'district', 'year', 'var', 'value', 'geometry']]

    return shapefile_merged
```


```python
test_df = plot_var_map(merged_df, de_nuts_gdf, var='wb', year=2023)
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nuts_id</th>
      <th>district_no</th>
      <th>NUTS_NAME</th>
      <th>district</th>
      <th>year</th>
      <th>var</th>
      <th>value</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DE11B</td>
      <td>8128.0</td>
      <td>Main-Tauber-Kreis</td>
      <td>Main-Tauber-Kreis</td>
      <td>2023.0</td>
      <td>wb</td>
      <td>6.15</td>
      <td>POLYGON ((1074230.536 6408356.046, 1073820.827...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DE11C</td>
      <td>8135.0</td>
      <td>Heidenheim</td>
      <td>Heidenheim, Landkreis</td>
      <td>2023.0</td>
      <td>wb</td>
      <td>NaN</td>
      <td>MULTIPOLYGON (((1131091.261 6235073.568, 11312...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DE11D</td>
      <td>8136.0</td>
      <td>Ostalbkreis</td>
      <td>Ostalbkreis</td>
      <td>2023.0</td>
      <td>wb</td>
      <td>7.19</td>
      <td>MULTIPOLYGON (((1141777.678 6284962.486, 11412...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DE121</td>
      <td>8211.0</td>
      <td>Baden-Baden, Stadtkreis</td>
      <td>Baden-Baden, kreisfreie Stadt</td>
      <td>2023.0</td>
      <td>wb</td>
      <td>NaN</td>
      <td>MULTIPOLYGON (((910859.613 6248068.047, 913127...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DE122</td>
      <td>8212.0</td>
      <td>Karlsruhe, Stadtkreis</td>
      <td>Karlsruhe, kreisfreie Stadt</td>
      <td>2023.0</td>
      <td>wb</td>
      <td>NaN</td>
      <td>POLYGON ((938225.711 6286986.826, 940668.057 6...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>450</th>
      <td>DE5</td>
      <td>NaN</td>
      <td>Bremen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MULTIPOLYGON (((949166.140 7023798.944, 952179...</td>
    </tr>
    <tr>
      <th>451</th>
      <td>DE6</td>
      <td>NaN</td>
      <td>Hamburg</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MULTIPOLYGON (((1134475.694 7117788.896, 11336...</td>
    </tr>
    <tr>
      <th>452</th>
      <td>DEE</td>
      <td>NaN</td>
      <td>Sachsen-Anhalt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MULTIPOLYGON (((1294568.737 6984897.844, 12963...</td>
    </tr>
    <tr>
      <th>453</th>
      <td>DE7</td>
      <td>NaN</td>
      <td>Hessen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MULTIPOLYGON (((1057294.287 6737363.291, 10568...</td>
    </tr>
    <tr>
      <th>454</th>
      <td>DE</td>
      <td>NaN</td>
      <td>Deutschland</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MULTIPOLYGON (((1163782.809 6033270.891, 11632...</td>
    </tr>
  </tbody>
</table>
<p>455 rows × 8 columns</p>
</div>




```python
merged_df.shape
```




    (187631, 8)




```python
merged_df['value'].isnull().sum()
```




    43290




```python
merged_df[merged_df['value']>0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district_no</th>
      <th>district</th>
      <th>nuts_id</th>
      <th>year</th>
      <th>var</th>
      <th>measure</th>
      <th>value</th>
      <th>outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>oats</td>
      <td>yield</td>
      <td>4.95</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>rye</td>
      <td>yield</td>
      <td>4.35</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>sb</td>
      <td>yield</td>
      <td>3.63</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>silage_maize</td>
      <td>yield</td>
      <td>43.60</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1001</td>
      <td>Flensburg, kreisfreie Stadt</td>
      <td>DEF01</td>
      <td>1979</td>
      <td>wb</td>
      <td>yield</td>
      <td>4.52</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>187630</th>
      <td>16077</td>
      <td>Altenburger Land, Landkreis</td>
      <td>DEG0M</td>
      <td>2023</td>
      <td>silage_maize</td>
      <td>yield</td>
      <td>41.94</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>186836</th>
      <td>16077</td>
      <td>Altenburger Land, Landkreis</td>
      <td>DEG0M</td>
      <td>2023</td>
      <td>sugarbeet</td>
      <td>yield</td>
      <td>76.63</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>184851</th>
      <td>16077</td>
      <td>Altenburger Land, Landkreis</td>
      <td>DEG0M</td>
      <td>2023</td>
      <td>wb</td>
      <td>yield</td>
      <td>10.04</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>187233</th>
      <td>16077</td>
      <td>Altenburger Land, Landkreis</td>
      <td>DEG0M</td>
      <td>2023</td>
      <td>wrape</td>
      <td>yield</td>
      <td>4.34</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>184057</th>
      <td>16077</td>
      <td>Altenburger Land, Landkreis</td>
      <td>DEG0M</td>
      <td>2023</td>
      <td>ww</td>
      <td>yield</td>
      <td>9.32</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>144341 rows × 8 columns</p>
</div>




```python
144341 + 43290
```




    187631




```python
test_df.plot(column='value', edgecolor='k', linewidth=0.3, figsize=(8, 8), legend=True)
```




    <Axes: >




    
![png](00_yield_data_preparation_files/00_yield_data_preparation_20_1.png)
    


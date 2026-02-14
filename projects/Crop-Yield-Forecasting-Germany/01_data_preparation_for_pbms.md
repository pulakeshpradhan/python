# **Data Preparation for PBMs**

## **Import Dependencies**


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
from glob import glob
import json
from tqdm.auto import tqdm
import ee
import geemap
from sklearn.neighbors import BallTree

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = 'Times New Roman'

import warnings
warnings.filterwarnings('ignore')

out_master_dir = r'datasets\master'
out_temp_dir = r'temp_data'
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



## **Instantiate a Map Object**


```python
# ee.Authenticate()
# ee.Initialize()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# Import Germany Shapefile
de_roi = ee.FeatureCollection('users/geonextgis/Germany_Administrative_Level_2')
poly_style = {'fillColor': '00000000', 'color': 'black', 'width': 1}
Map = geemap.Map(basemap='Esri.WorldImagery')
Map.addLayer(de_roi.style(**poly_style), {}, 'DE ROI')
Map.centerObject(de_roi, 6)
Map
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    Map(center=[51.055719127031935, 10.373828310619029], controls=(WidgetControl(options=['position', 'transparent…



## **Read the Datasets**


```python
# Read the DE NUTS3 Shapefile
de_nuts3_gdf = gpd.read_file("datasets\shapefiles\DE_NUTS\DE_NUTS_3.shp")
de_nuts3_gdf = de_nuts3_gdf[de_nuts3_gdf['LEVL_CODE']==3] # Filter for NUTS3
print(de_nuts3_gdf.shape)
de_nuts3_gdf.head()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    (400, 6)
    




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
      <td>POLYGON ((1074230.536 6408356.046, 1073820.827...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DE11C</td>
      <td>3</td>
      <td>DE</td>
      <td>Heidenheim</td>
      <td>Heidenheim</td>
      <td>MULTIPOLYGON (((1131091.261 6235073.568, 11312...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DE11D</td>
      <td>3</td>
      <td>DE</td>
      <td>Ostalbkreis</td>
      <td>Ostalbkreis</td>
      <td>MULTIPOLYGON (((1141777.678 6284962.486, 11412...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DE121</td>
      <td>3</td>
      <td>DE</td>
      <td>Baden-Baden, Stadtkreis</td>
      <td>Baden-Baden, Stadtkreis</td>
      <td>MULTIPOLYGON (((910859.613 6248068.047, 913127...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DE122</td>
      <td>3</td>
      <td>DE</td>
      <td>Karlsruhe, Stadtkreis</td>
      <td>Karlsruhe, Stadtkreis</td>
      <td>POLYGON ((938225.711 6286986.826, 940668.057 6...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Read the soil coordinates
soil_coords_df = pd.read_csv("datasets\csvs\Site_Soil_BZE_WGS84_Coords.csv")

# Read the soil attributes
soil_attributes_df = pd.read_excel(
    "datasets\csvs\SoilData.xlsx",
    sheet_name='SoilData'
)
soil_attributes_df.drop(columns=['Lon', 'Lat'], inplace=True)
soil_attributes_df.rename(columns={'Location_id': 'PointID'}, inplace=True)

print('Soil coordinates data shape:', soil_coords_df.shape)
print('Soil coordinates attributes shape:', soil_attributes_df.shape)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    Soil coordinates data shape: (3104, 3)
    Soil coordinates attributes shape: (3087, 31)
    


```python
# Read the ESA WorldCover dataset
esa_lulc = ee.ImageCollection("ESA/WorldCover/v100").first()
visualization = {
  'bands': ['Map'],
}
Map.addLayer(esa_lulc, visualization, 'ESA LULC', opacity=0.6)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



## **Soil Data Preparation**


```python
# Convert the soil coordinates into geodataframe
soil_coords_geometry = gpd.points_from_xy(soil_coords_df['Longitude'], soil_coords_df['Latitude'], crs=4326)
soil_coords_gdf = gpd.GeoDataFrame(soil_coords_df, geometry=soil_coords_geometry).to_crs(epsg=3857)
print(soil_coords_gdf.shape)
soil_coords_gdf.head()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    (3104, 4)
    




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
      <th>PointID</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>8.411608</td>
      <td>54.859923</td>
      <td>POINT (936375.919 7334727.347)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>8.697143</td>
      <td>54.864382</td>
      <td>POINT (968161.53 7335589.787)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>8.765298</td>
      <td>54.866979</td>
      <td>POINT (975748.51 7336092.132)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>8.959032</td>
      <td>54.863920</td>
      <td>POINT (997314.88 7335500.425)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>9.078456</td>
      <td>54.870685</td>
      <td>POINT (1010609.099 7336809.049)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge the NUTS3 information in the soil coordinates dataframe
soil_coords_gdf = soil_coords_gdf.sjoin(de_nuts3_gdf[['NUTS_ID', 'NUTS_NAME', 'geometry']], how='left', predicate='intersects')
print(soil_coords_gdf.shape)
soil_coords_gdf.head()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    (3104, 7)
    




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
      <th>PointID</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>geometry</th>
      <th>index_right</th>
      <th>NUTS_ID</th>
      <th>NUTS_NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>8.411608</td>
      <td>54.859923</td>
      <td>POINT (936375.919 7334727.347)</td>
      <td>166.0</td>
      <td>DEF07</td>
      <td>Nordfriesland</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>8.697143</td>
      <td>54.864382</td>
      <td>POINT (968161.53 7335589.787)</td>
      <td>166.0</td>
      <td>DEF07</td>
      <td>Nordfriesland</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>8.765298</td>
      <td>54.866979</td>
      <td>POINT (975748.51 7336092.132)</td>
      <td>166.0</td>
      <td>DEF07</td>
      <td>Nordfriesland</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>8.959032</td>
      <td>54.863920</td>
      <td>POINT (997314.88 7335500.425)</td>
      <td>166.0</td>
      <td>DEF07</td>
      <td>Nordfriesland</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>9.078456</td>
      <td>54.870685</td>
      <td>POINT (1010609.099 7336809.049)</td>
      <td>166.0</td>
      <td>DEF07</td>
      <td>Nordfriesland</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge the soil coordinate information in the soil attributes dataframe
soil_attributes_gdf = pd.merge(
    left=soil_coords_gdf[['PointID', 'Longitude', 'Latitude', 'NUTS_ID', 'NUTS_NAME', 'geometry']], 
    right=soil_attributes_df, 
    on='PointID', 
    how='inner')

soil_attributes_gdf.columns = [col.replace('.0', '') for col in soil_attributes_gdf.columns]
soil_attributes_gdf.to_crs(crs='epsg:31467', inplace=True) # change the CRS
soil_attributes_ee = geemap.gdf_to_ee(soil_attributes_gdf, geodesic=False)

soil_style = {'color': 'green', 'pointSize': 5}
Map.addLayer(soil_attributes_ee.style(**soil_style), {}, 'Soil Points')

print(soil_attributes_gdf.shape)
soil_attributes_gdf.head()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    (3087, 36)
    




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
      <th>PointID</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>NUTS_ID</th>
      <th>NUTS_NAME</th>
      <th>geometry</th>
      <th>SoilLayerDepth_10cm</th>
      <th>SoilLayerDepth_30cm</th>
      <th>SoilLayerDepth_50cm</th>
      <th>SoilLayerDepth_70cm</th>
      <th>...</th>
      <th>BD_50cm</th>
      <th>BD_70cm</th>
      <th>BD_1m</th>
      <th>BD_2m</th>
      <th>OC_10cm</th>
      <th>OC_30cm</th>
      <th>OC_50cm</th>
      <th>OC_70cm</th>
      <th>OC_1m</th>
      <th>OC_2m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>8.411608</td>
      <td>54.859923</td>
      <td>DEF07</td>
      <td>Nordfriesland</td>
      <td>POINT (3462285.321 6081355.028)</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>...</td>
      <td>1.45</td>
      <td>1.30</td>
      <td>1.54</td>
      <td>1.54</td>
      <td>2.183</td>
      <td>2.075</td>
      <td>1.151</td>
      <td>0.691</td>
      <td>0.157</td>
      <td>0.157</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>8.697143</td>
      <td>54.864382</td>
      <td>DEF07</td>
      <td>Nordfriesland</td>
      <td>POINT (3480623.416 6081734.949)</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>...</td>
      <td>1.57</td>
      <td>1.39</td>
      <td>1.47</td>
      <td>1.47</td>
      <td>2.064</td>
      <td>1.589</td>
      <td>0.574</td>
      <td>0.579</td>
      <td>0.581</td>
      <td>0.581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>8.765298</td>
      <td>54.866979</td>
      <td>DEF07</td>
      <td>Nordfriesland</td>
      <td>POINT (3485000.58 6082007.3)</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>...</td>
      <td>1.37</td>
      <td>1.48</td>
      <td>1.49</td>
      <td>1.49</td>
      <td>1.915</td>
      <td>1.517</td>
      <td>2.220</td>
      <td>1.028</td>
      <td>0.652</td>
      <td>0.652</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>8.959032</td>
      <td>54.863920</td>
      <td>DEF07</td>
      <td>Nordfriesland</td>
      <td>POINT (3497439.177 6081642.411)</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>...</td>
      <td>1.45</td>
      <td>1.48</td>
      <td>1.60</td>
      <td>1.60</td>
      <td>3.375</td>
      <td>1.701</td>
      <td>1.233</td>
      <td>1.089</td>
      <td>0.674</td>
      <td>0.674</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>9.078456</td>
      <td>54.870685</td>
      <td>DEF07</td>
      <td>Nordfriesland</td>
      <td>POINT (3505106.596 6082397.618)</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>...</td>
      <td>1.42</td>
      <td>0.89</td>
      <td>1.35</td>
      <td>1.35</td>
      <td>2.447</td>
      <td>1.166</td>
      <td>0.398</td>
      <td>3.570</td>
      <td>0.241</td>
      <td>0.241</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
# Save the Soil data
# soil_attributes_gdf.to_csv(os.path.join(out_csv_dir, 'DE_Soil_BZE_Master.csv'), index=False)
# soil_attributes_gdf.to_file(os.path.join(out_master_dir, 'DE_Soil_BZE_Master.shp'), index=False)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



## **Climate Grid Preparation**


```python
# Read the DWD Climate JSON file
dwd_json_path = "datasets\shapefiles\dwd_ubn_latlon_to_rowcol.json"

with open(dwd_json_path, 'r') as file:
    dwd_json = json.load(file)

# Function to convert the json data into a table
def json_to_table(data):
    coords_info = [i[0] for i in dwd_json]
    row_col_info = [i[1] for i in dwd_json]

    coords_info = pd.DataFrame(coords_info, columns=['Latitude', 'Longitude'])
    row_col_info = pd.DataFrame(row_col_info, columns=['Row', 'Column'])

    final_df = pd.concat((coords_info, row_col_info), axis=1)

    return final_df

dwd_grid_df = json_to_table(dwd_json)
print(dwd_grid_df.shape)
dwd_grid_df.head()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    (358303, 4)
    




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
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Row</th>
      <th>Column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>55.054328</td>
      <td>8.402969</td>
      <td>0</td>
      <td>181</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.054404</td>
      <td>8.418616</td>
      <td>0</td>
      <td>182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.045268</td>
      <td>8.387459</td>
      <td>1</td>
      <td>180</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55.045346</td>
      <td>8.403102</td>
      <td>1</td>
      <td>181</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.036286</td>
      <td>8.387596</td>
      <td>2</td>
      <td>180</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert the data into geodataframe
dwd_grid_gdf = gpd.GeoDataFrame(
    dwd_grid_df, 
    geometry=gpd.points_from_xy(dwd_grid_df['Longitude'], dwd_grid_df['Latitude']),
    crs=4326)
print(dwd_grid_gdf.shape)
dwd_grid_gdf.head()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    (358303, 5)
    




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
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Row</th>
      <th>Column</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>55.054328</td>
      <td>8.402969</td>
      <td>0</td>
      <td>181</td>
      <td>POINT (8.40297 55.05433)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.054404</td>
      <td>8.418616</td>
      <td>0</td>
      <td>182</td>
      <td>POINT (8.41862 55.0544)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.045268</td>
      <td>8.387459</td>
      <td>1</td>
      <td>180</td>
      <td>POINT (8.38746 55.04527)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55.045346</td>
      <td>8.403102</td>
      <td>1</td>
      <td>181</td>
      <td>POINT (8.4031 55.04535)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.036286</td>
      <td>8.387596</td>
      <td>2</td>
      <td>180</td>
      <td>POINT (8.3876 55.03629)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# # Save the data
# dwd_grid_df.to_csv(os.path.join(out_csv_dir, 'DE_DWD_UBN_Centroids.csv'), index=False)

# dwd_grid_gdf.to_crs(31467).to_file("datasets\shapefiles\DE_DWD_UBN_GRIDS\DE_DWD_UBN_Centroids_EPSG_31467.shp")
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



## **Filter Points falling on Cropland**


```python
# Load the DWD grid centroids on EE Map object
dwd_centroids_ee = ee.FeatureCollection('projects/ee-geonextgis/assets/DE_DWD_UBN_Centroids_EPSG_31467') 
dwd_style = {
    'color': 'red',
    'pointSize': 1,
    'width': 1,
    'fillColor': '00000000'
}

# Create a 500m buffer for each feature and get bounds
dwd_grids_ee = dwd_centroids_ee.map(lambda f: f.buffer(500).bounds())

Map.addLayer(dwd_centroids_ee.style(**dwd_style), {}, 'DWD Centroids', False)
Map.addLayer(dwd_grids_ee.style(**dwd_style), {}, 'DWD Grids', False)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# # Iterate over the columns and extract the data in the temporary folder
# for column_id in tqdm(sorted(dwd_grid_gdf['Column'].unique())):

#     filered_column_cells = dwd_grids_ee.filter(ee.Filter.eq('Column', int(column_id)))

#     try:
#         out_data_path = os.path.join(out_temp_dir, f'Col_{column_id}_DWD_LULC.csv')

#         # Extract LULC info for all the DWD cells
#         dwd_lulc_zonal_stat = geemap.zonal_statistics_by_group(
#             esa_lulc, filered_column_cells, out_data_path, statistics_type='SUM'
#         )

#         print(f'Column ID: {column_id} | Data saved at {out_data_path}')

#     except:
#         print(f'Column ID: {column_id} | Error.')
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# Merge all the data in a single file
concatenated_df = pd.DataFrame()

lulc_class_columns = ['Class_10', 'Class_20', 'Class_30', 'Class_40', 'Class_50', 'Class_60',
                      'Class_70', 'Class_80', 'Class_90', 'Class_95', 'Class_100']

temp_file_paths = glob(out_temp_dir + '\\*.csv')

for path in tqdm(temp_file_paths):
    temp_df = pd.read_csv(path)

    for col in lulc_class_columns:
        if col not in list(temp_df.columns):
            temp_df[col] = 0

        temp_df[col] = np.round((temp_df[col] / temp_df['Class_sum']) * 100, 4)

    temp_df = temp_df[['Row', 'Column', 'Longitude', 'Latitude'] + lulc_class_columns]

    concatenated_df = pd.concat((concatenated_df, temp_df), axis=0)

# Filter the grid cell where cropland ('Class_40') area is more than 20%
dwd_cropland_df = concatenated_df[concatenated_df['Class_40']>=20].iloc[:, :4]
dwd_cropland_gdf = pd.merge(left=dwd_cropland_df, right=dwd_grid_gdf[['Row', 'Column', 'geometry']], on=['Row', 'Column'], how='left')
dwd_cropland_gdf = gpd.GeoDataFrame(dwd_cropland_gdf)
print(dwd_cropland_gdf.shape)
dwd_cropland_gdf.head()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




      0%|          | 0/640 [00:00<?, ?it/s]


    (190658, 5)
    




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
      <th>Row</th>
      <th>Column</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>444</td>
      <td>0</td>
      <td>5.876024</td>
      <td>51.024361</td>
      <td>POINT (5.87602 51.02436)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>477</td>
      <td>100</td>
      <td>7.311274</td>
      <td>50.757258</td>
      <td>POINT (7.31127 50.75726)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>481</td>
      <td>100</td>
      <td>7.312566</td>
      <td>50.721317</td>
      <td>POINT (7.31257 50.72132)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>482</td>
      <td>100</td>
      <td>7.312888</td>
      <td>50.712331</td>
      <td>POINT (7.31289 50.71233)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>511</td>
      <td>100</td>
      <td>7.322172</td>
      <td>50.451747</td>
      <td>POINT (7.32217 50.45175)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add the NUTS information
dwd_cropland_gdf = dwd_cropland_gdf.to_crs(crs='epsg:3857')
dwd_cropland_gdf = dwd_cropland_gdf.sjoin(de_nuts3_gdf[['NUTS_ID', 'NUTS_NAME', 'geometry']], how='left', predicate='intersects')
dwd_cropland_gdf.dropna(inplace=True)
dwd_cropland_gdf.sort_values(by=['NUTS_ID'], inplace=True)
dwd_cropland_gdf.reset_index(drop=True, inplace=True)
dwd_cropland_gdf['Cell_ID'] = dwd_cropland_gdf.index
dwd_cropland_gdf = dwd_cropland_gdf[['Cell_ID', 'Row', 'Column', 'Latitude', 'Longitude', 'NUTS_ID', 'NUTS_NAME', 'geometry']]
print(dwd_cropland_gdf.shape)
dwd_cropland_gdf.head()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    (190364, 8)
    




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
      <th>Cell_ID</th>
      <th>Row</th>
      <th>Column</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>NUTS_ID</th>
      <th>NUTS_NAME</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>703</td>
      <td>234</td>
      <td>48.737371</td>
      <td>9.201742</td>
      <td>DE111</td>
      <td>Stuttgart, Stadtkreis</td>
      <td>POINT (1024333.233 6230415.593)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>691</td>
      <td>236</td>
      <td>48.845227</td>
      <td>9.229425</td>
      <td>DE111</td>
      <td>Stuttgart, Stadtkreis</td>
      <td>POINT (1027414.872 6248640.293)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>707</td>
      <td>233</td>
      <td>48.701424</td>
      <td>9.188012</td>
      <td>DE111</td>
      <td>Stuttgart, Stadtkreis</td>
      <td>POINT (1022804.862 6224350.32)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>706</td>
      <td>233</td>
      <td>48.710416</td>
      <td>9.188046</td>
      <td>DE111</td>
      <td>Stuttgart, Stadtkreis</td>
      <td>POINT (1022808.604 6225867.209)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>705</td>
      <td>233</td>
      <td>48.719409</td>
      <td>9.188080</td>
      <td>DE111</td>
      <td>Stuttgart, Stadtkreis</td>
      <td>POINT (1022812.348 6227384.366)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Save the data
# dwd_cropland_gdf.to_csv(os.path.join(out_master_dir, 'DE_DWD_UBN_Crop.csv'), index=False)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



## **Find k-Nearest Soil Points for Each Climate Grid Cell**


```python
# Convert the soil data in the same coordinate system
soil_attributes_gdf.to_crs(crs='epsg:3857', inplace=True)
soil_attributes_gdf.crs == dwd_cropland_gdf.crs
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    True




```python
# Extract coordinates
dwd_cropland_coords = np.array(list(zip(dwd_cropland_gdf.geometry.x, dwd_cropland_gdf.geometry.y)))
soil_coords = np.array(list(zip(soil_attributes_gdf.geometry.x, soil_attributes_gdf.geometry.y)))

# Build BallTree for fast nearest neighbor search
tree = BallTree(soil_coords, metric='euclidean')

# Define number of neighbors (k)
k = 5  # Adjust based on data density

# Query k-nearest neighbors for each climate point
distances, indices = tree.query(dwd_cropland_coords, k=k)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



## **Compute Inverse Distance Weighted (IDW) Average**


```python
# Define soil properties to interpolate
soil_properties = soil_attributes_gdf.columns[12:]  # Add all required properties
power = 2  # IDW power parameter

# Dictionary to store weighted values for each property
weighted_results = {prop: [] for prop in soil_properties}

# Compute IDW for each climate grid cell
for i, climate_point in tqdm(enumerate(dwd_cropland_gdf.geometry)):
    nearest_soil_points = soil_attributes_gdf.iloc[indices[i]]  # Get k nearest neighbors
    nearest_distances = distances[i]

    # Avoid division by zero
    nearest_distances[nearest_distances == 0] = 1e-6

    # Compute weights (w = 1/d^p)
    weights = 1 / (nearest_distances ** power)

    # Compute weighted average for each soil property
    for prop in soil_properties:
        weighted_avg = np.round(np.sum(weights * nearest_soil_points[prop].values) / np.sum(weights), 4)
        weighted_results[prop].append(weighted_avg)

# Assign weighted values to climate GeoDataFrame
for prop in soil_properties:
    dwd_cropland_gdf[f"{prop}"] = weighted_results[prop]
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




    0it [00:00, ?it/s]



```python
# Add soil layer depth columns (needed for PBMs)
dwd_cropland_gdf['SoilLayerDepth_10cm'] = 0.1
dwd_cropland_gdf['SoilLayerDepth_30cm'] = 0.3
dwd_cropland_gdf['SoilLayerDepth_50cm'] = 0.5
dwd_cropland_gdf['SoilLayerDepth_70cm'] = 0.7
dwd_cropland_gdf['SoilLayerDepth_1m'] = 1
dwd_cropland_gdf['SoilLayerDepth_2m'] = 2

# Reorder the columns
dwd_cropland_gdf = dwd_cropland_gdf[list(dwd_cropland_gdf.columns[:7]) + list(soil_attributes_gdf.columns[6:12]) + list(soil_properties) + ['geometry']]
print(dwd_cropland_gdf.shape)
dwd_cropland_gdf.head()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    (190364, 38)
    




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
      <th>Cell_ID</th>
      <th>Row</th>
      <th>Column</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>NUTS_ID</th>
      <th>NUTS_NAME</th>
      <th>SoilLayerDepth_10cm</th>
      <th>SoilLayerDepth_30cm</th>
      <th>SoilLayerDepth_50cm</th>
      <th>...</th>
      <th>BD_70cm</th>
      <th>BD_1m</th>
      <th>BD_2m</th>
      <th>OC_10cm</th>
      <th>OC_30cm</th>
      <th>OC_50cm</th>
      <th>OC_70cm</th>
      <th>OC_1m</th>
      <th>OC_2m</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>703</td>
      <td>234</td>
      <td>48.737371</td>
      <td>9.201742</td>
      <td>DE111</td>
      <td>Stuttgart, Stadtkreis</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>1.5407</td>
      <td>1.5394</td>
      <td>1.5394</td>
      <td>1.2970</td>
      <td>1.1749</td>
      <td>0.3660</td>
      <td>0.3316</td>
      <td>0.1847</td>
      <td>0.1847</td>
      <td>POINT (1024333.233 6230415.593)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>691</td>
      <td>236</td>
      <td>48.845227</td>
      <td>9.229425</td>
      <td>DE111</td>
      <td>Stuttgart, Stadtkreis</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>1.4462</td>
      <td>1.4607</td>
      <td>1.4607</td>
      <td>2.2463</td>
      <td>1.8119</td>
      <td>0.6600</td>
      <td>0.5122</td>
      <td>0.2873</td>
      <td>0.2873</td>
      <td>POINT (1027414.872 6248640.293)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>707</td>
      <td>233</td>
      <td>48.701424</td>
      <td>9.188012</td>
      <td>DE111</td>
      <td>Stuttgart, Stadtkreis</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>1.5569</td>
      <td>1.4867</td>
      <td>1.4867</td>
      <td>1.3170</td>
      <td>1.1219</td>
      <td>0.3438</td>
      <td>0.3209</td>
      <td>0.1713</td>
      <td>0.1713</td>
      <td>POINT (1022804.862 6224350.32)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>706</td>
      <td>233</td>
      <td>48.710416</td>
      <td>9.188046</td>
      <td>DE111</td>
      <td>Stuttgart, Stadtkreis</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>1.5501</td>
      <td>1.5007</td>
      <td>1.5007</td>
      <td>1.2979</td>
      <td>1.1741</td>
      <td>0.3449</td>
      <td>0.3250</td>
      <td>0.1761</td>
      <td>0.1761</td>
      <td>POINT (1022808.604 6225867.209)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>705</td>
      <td>233</td>
      <td>48.719409</td>
      <td>9.188080</td>
      <td>DE111</td>
      <td>Stuttgart, Stadtkreis</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>1.5484</td>
      <td>1.5136</td>
      <td>1.5136</td>
      <td>1.3000</td>
      <td>1.1741</td>
      <td>0.3491</td>
      <td>0.3254</td>
      <td>0.1780</td>
      <td>0.1780</td>
      <td>POINT (1022812.348 6227384.366)</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
# Save the data
# dwd_cropland_gdf.drop(columns='geometry').to_csv(os.path.join(out_master_dir, 'DE_DWD_UBN_Crop_Soil.csv'), index=False)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/Crop-Yield-Forecasting-Germany/02_data_preparation_for_pbms.ipynb)

# **Data Preparation for PBMs - 2**

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

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = 'Times New Roman'

import warnings
warnings.filterwarnings('ignore')

out_master_dir = r'datasets\master'
out_temp_dir = r'temp_data'
```

## **Read the Datasets**


```python
# Read the Soil Hydraulic Property dataset
pbm_data = pd.read_csv('datasets\csvs\soilhydraulic_property_Germany_Points_Amit.csv', delimiter=';')
pbm_data = pbm_data.iloc[:, :-2]
print(pbm_data.shape)
pbm_data.head()
```

    (190364, 144)
    




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
      <th>location</th>
      <th>dampingdepth</th>
      <th>soilwater_fc_global</th>
      <th>soilwater_sat_global</th>
      <th>drainage_rate</th>
      <th>deltatheta</th>
      <th>DZF</th>
      <th>depth_1</th>
      <th>depth_2</th>
      <th>depth_3</th>
      <th>...</th>
      <th>InitialFixedPConcentration_6</th>
      <th>slimalfa_1</th>
      <th>slimalfa_2</th>
      <th>slimalfa_3</th>
      <th>slimalfa_4</th>
      <th>slimalfa_5</th>
      <th>slimalfa_6</th>
      <th>Nitrogen</th>
      <th>Phosphorous</th>
      <th>Potassium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>0.305742</td>
      <td>0.41441</td>
      <td>50</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>57</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>120.45755</td>
      <td>7.1606</td>
      <td>16.997152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>0.352146</td>
      <td>0.447852</td>
      <td>50</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>57</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>120.45755</td>
      <td>7.1606</td>
      <td>16.997152</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>6</td>
      <td>0.292897</td>
      <td>0.422044</td>
      <td>50</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>57</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>120.45755</td>
      <td>7.1606</td>
      <td>16.997152</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6</td>
      <td>0.300213</td>
      <td>0.420899</td>
      <td>50</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>57</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>120.45755</td>
      <td>7.1606</td>
      <td>16.997152</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6</td>
      <td>0.30233</td>
      <td>0.418748</td>
      <td>50</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>57</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>120.45755</td>
      <td>7.1606</td>
      <td>16.997152</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 144 columns</p>
</div>



## **Data Processing**


```python
# Divide the Nitrogen, Phosphorous, and Potassium data by 10
pbm_data[['Nitrogen', 'Phosphorous', 'Potassium']] = pbm_data[['Nitrogen', 'Phosphorous', 'Potassium']] / 10
print(pbm_data.shape)
pbm_data.head()
```

    (190364, 144)
    




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
      <th>location</th>
      <th>dampingdepth</th>
      <th>soilwater_fc_global</th>
      <th>soilwater_sat_global</th>
      <th>drainage_rate</th>
      <th>deltatheta</th>
      <th>DZF</th>
      <th>depth_1</th>
      <th>depth_2</th>
      <th>depth_3</th>
      <th>...</th>
      <th>InitialFixedPConcentration_6</th>
      <th>slimalfa_1</th>
      <th>slimalfa_2</th>
      <th>slimalfa_3</th>
      <th>slimalfa_4</th>
      <th>slimalfa_5</th>
      <th>slimalfa_6</th>
      <th>Nitrogen</th>
      <th>Phosphorous</th>
      <th>Potassium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>0.305742</td>
      <td>0.41441</td>
      <td>50</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>57</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>12.045755</td>
      <td>0.71606</td>
      <td>1.699715</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>0.352146</td>
      <td>0.447852</td>
      <td>50</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>57</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>12.045755</td>
      <td>0.71606</td>
      <td>1.699715</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>6</td>
      <td>0.292897</td>
      <td>0.422044</td>
      <td>50</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>57</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>12.045755</td>
      <td>0.71606</td>
      <td>1.699715</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6</td>
      <td>0.300213</td>
      <td>0.420899</td>
      <td>50</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>57</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>12.045755</td>
      <td>0.71606</td>
      <td>1.699715</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6</td>
      <td>0.30233</td>
      <td>0.418748</td>
      <td>50</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>...</td>
      <td>57</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>12.045755</td>
      <td>0.71606</td>
      <td>1.699715</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 144 columns</p>
</div>




```python
# Define the scenarios
def process_data(row, fertilizer_scenario=2, crop_cyle_count=0, crop='winter wheat'):

    location_values = [row['location']] * 6 
    fertilizer_scenarios = [fertilizer_scenario] * 6
    crop_cyle_counts = [crop_cyle_count] * 6
    crop_values = [crop] * 6
    type_values = ['PTotal', 'KTotal', 'NTotal'] * 2
    dvs_values = [0.001, 0.001, 0.25, 0.4, 0.4, 0.9]
    event_values = [1, 2, 3, 4, 5, 6]
    fertilizer_value = [round(float(v), 6) for v in [row['Phosphorous'], row['Potassium'], row['Nitrogen']]]
    fertilizer_values =  [round((v/2), 6) for v  in (fertilizer_value * 2)]

    final_df = pd.DataFrame({
        'location': location_values,
        'FertilizerScenario': fertilizer_scenarios,
        'CropCycleCount': crop_cyle_counts,
        'crop': crop_values,
        'Event': event_values,
        'vType': type_values,
        'DVS': dvs_values,
        'Amount': fertilizer_values
    })

    return final_df
```


```python
# Apply the algorithm on each rows
pbm_data_processed = pbm_data.apply(process_data, axis=1)
pbm_data_processed = pd.concat(pbm_data_processed.tolist(), ignore_index=True)
print(pbm_data_processed.shape)
pbm_data_processed.head()
```

    (1142184, 8)
    




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
      <th>location</th>
      <th>FertilizerScenario</th>
      <th>CropCycleCount</th>
      <th>crop</th>
      <th>Event</th>
      <th>vType</th>
      <th>DVS</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>winter wheat</td>
      <td>1</td>
      <td>PTotal</td>
      <td>0.001</td>
      <td>0.358030</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>winter wheat</td>
      <td>2</td>
      <td>KTotal</td>
      <td>0.001</td>
      <td>0.849858</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>winter wheat</td>
      <td>3</td>
      <td>NTotal</td>
      <td>0.250</td>
      <td>6.022877</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>winter wheat</td>
      <td>4</td>
      <td>PTotal</td>
      <td>0.400</td>
      <td>0.358030</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>winter wheat</td>
      <td>5</td>
      <td>KTotal</td>
      <td>0.400</td>
      <td>0.849858</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Save the data 
# pbm_data_processed.to_csv(os.path.join(out_master_dir, 'fertilizer_Soil3_AllKreis_Krishna.csv'), index=False)
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
path = "D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\significant_earthquakes_2000_2020.csv"
```


```python
df = pd.read_csv(path)
df.head()
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
      <th>Year</th>
      <th>Mo</th>
      <th>Dy</th>
      <th>Location Name</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Focal Depth (km)</th>
      <th>Mag</th>
      <th>Total Deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>1</td>
      <td>3</td>
      <td>INDIA-BANGLADESH BORDER:  MAHESHKHALI</td>
      <td>22.132</td>
      <td>92.771</td>
      <td>33.0</td>
      <td>4.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000</td>
      <td>1</td>
      <td>11</td>
      <td>CHINA:  LIAONING PROVINCE</td>
      <td>40.498</td>
      <td>122.994</td>
      <td>10.0</td>
      <td>5.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>1</td>
      <td>14</td>
      <td>CHINA:  YUNNAN PROVINCE:  YAOAN COUNTY</td>
      <td>25.607</td>
      <td>101.063</td>
      <td>33.0</td>
      <td>5.9</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>2</td>
      <td>2</td>
      <td>IRAN:  BARDASKAN, KASHMAR</td>
      <td>35.288</td>
      <td>58.218</td>
      <td>33.0</td>
      <td>5.3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000</td>
      <td>2</td>
      <td>7</td>
      <td>SOUTH AFRICA; SWAZILAND:  MBABANE-MANZINI</td>
      <td>-26.288</td>
      <td>30.888</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (1206, 9)




```python
df["Mag"].max()
```




    9.1




```python
df["Location Name"][df["Total Deaths"] > 50000]
```




    272         INDONESIA:  SUMATRA:  ACEH:  OFF WEST COAST
    320    PAKISTAN:  MUZAFFARABAD, URI, ANANTNAG, BARAMULA
    490                            CHINA:  SICHUAN PROVINCE
    607                              HAITI:  PORT-AU-PRINCE
    Name: Location Name, dtype: object




```python
df["Location Name"][df["Mag"] > 8.5]
```




    272    INDONESIA:  SUMATRA:  ACEH:  OFF WEST COAST
    294                      INDONESIA:  SUMATERA:  SW
    614          CHILE:  MAULE, CONCEPCION, TALCAHUANO
    674                                 JAPAN:  HONSHU
    736         INDONESIA:  N SUMATRA:  OFF WEST COAST
    Name: Location Name, dtype: object




```python
df["Mag"].mean()
```




    5.945054031587698




```python
df.columns
```




    Index(['Year', 'Mo', 'Dy', 'Location Name', 'Latitude', 'Longitude',
           'Focal Depth (km)', 'Mag', 'Total Deaths'],
          dtype='object')




```python
df.fillna(0, inplace=True)
df.head()
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
      <th>Year</th>
      <th>Mo</th>
      <th>Dy</th>
      <th>Location Name</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Focal Depth (km)</th>
      <th>Mag</th>
      <th>Total Deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>1</td>
      <td>3</td>
      <td>INDIA-BANGLADESH BORDER:  MAHESHKHALI</td>
      <td>22.132</td>
      <td>92.771</td>
      <td>33.0</td>
      <td>4.6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000</td>
      <td>1</td>
      <td>11</td>
      <td>CHINA:  LIAONING PROVINCE</td>
      <td>40.498</td>
      <td>122.994</td>
      <td>10.0</td>
      <td>5.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>1</td>
      <td>14</td>
      <td>CHINA:  YUNNAN PROVINCE:  YAOAN COUNTY</td>
      <td>25.607</td>
      <td>101.063</td>
      <td>33.0</td>
      <td>5.9</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>2</td>
      <td>2</td>
      <td>IRAN:  BARDASKAN, KASHMAR</td>
      <td>35.288</td>
      <td>58.218</td>
      <td>33.0</td>
      <td>5.3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000</td>
      <td>2</td>
      <td>7</td>
      <td>SOUTH AFRICA; SWAZILAND:  MBABANE-MANZINI</td>
      <td>-26.288</td>
      <td>30.888</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_df = df[["Location Name", "Mag", "Total Deaths"]]
new_df
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
      <th>Location Name</th>
      <th>Mag</th>
      <th>Total Deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INDIA-BANGLADESH BORDER:  MAHESHKHALI</td>
      <td>4.6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CHINA:  LIAONING PROVINCE</td>
      <td>5.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CHINA:  YUNNAN PROVINCE:  YAOAN COUNTY</td>
      <td>5.9</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IRAN:  BARDASKAN, KASHMAR</td>
      <td>5.3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SOUTH AFRICA; SWAZILAND:  MBABANE-MANZINI</td>
      <td>4.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1201</th>
      <td>PHILIPPINES:  MASBATE</td>
      <td>6.6</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1202</th>
      <td>ALASKA</td>
      <td>7.6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1203</th>
      <td>GREECE:  SAMOS; TURKEY:  IZMIR</td>
      <td>7.0</td>
      <td>118.0</td>
    </tr>
    <tr>
      <th>1204</th>
      <td>CHILE:  OFF COAST CENTRAL</td>
      <td>6.7</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1205</th>
      <td>BALKANS NW:  CROATIA:  PETRINJA</td>
      <td>6.4</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
<p>1206 rows × 3 columns</p>
</div>




```python
new_df.describe()
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
      <th>Mag</th>
      <th>Total Deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1206.000000</td>
      <td>1206.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.930265</td>
      <td>681.509121</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.099304</td>
      <td>11757.576367</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.200000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.900000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.700000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.100000</td>
      <td>316000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.boxplot(new_df["Mag"])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x28fd619e1c0>,
      <matplotlib.lines.Line2D at 0x28fd619e460>],
     'caps': [<matplotlib.lines.Line2D at 0x28fd619e700>,
      <matplotlib.lines.Line2D at 0x28fd619e9a0>],
     'boxes': [<matplotlib.lines.Line2D at 0x28fd617eee0>],
     'medians': [<matplotlib.lines.Line2D at 0x28fd619ec40>],
     'fliers': [<matplotlib.lines.Line2D at 0x28fd619eee0>],
     'means': []}




    
![png](Untitled1-checkpoint_files/Untitled1-checkpoint_12_1.png)
    


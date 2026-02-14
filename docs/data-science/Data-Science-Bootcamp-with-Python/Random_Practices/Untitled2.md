[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/Random_Practices/Untitled2.ipynb)

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
df[df["Mag"] > 8.5]
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
      <th>272</th>
      <td>2004</td>
      <td>12</td>
      <td>26</td>
      <td>INDONESIA:  SUMATRA:  ACEH:  OFF WEST COAST</td>
      <td>3.316</td>
      <td>95.854</td>
      <td>30.0</td>
      <td>9.1</td>
      <td>227899.0</td>
    </tr>
    <tr>
      <th>294</th>
      <td>2005</td>
      <td>3</td>
      <td>28</td>
      <td>INDONESIA:  SUMATERA:  SW</td>
      <td>2.085</td>
      <td>97.108</td>
      <td>30.0</td>
      <td>8.6</td>
      <td>1313.0</td>
    </tr>
    <tr>
      <th>614</th>
      <td>2010</td>
      <td>2</td>
      <td>27</td>
      <td>CHILE:  MAULE, CONCEPCION, TALCAHUANO</td>
      <td>-36.122</td>
      <td>-72.898</td>
      <td>23.0</td>
      <td>8.8</td>
      <td>558.0</td>
    </tr>
    <tr>
      <th>674</th>
      <td>2011</td>
      <td>3</td>
      <td>11</td>
      <td>JAPAN:  HONSHU</td>
      <td>38.297</td>
      <td>142.372</td>
      <td>30.0</td>
      <td>9.1</td>
      <td>18428.0</td>
    </tr>
    <tr>
      <th>736</th>
      <td>2012</td>
      <td>4</td>
      <td>11</td>
      <td>INDONESIA:  N SUMATRA:  OFF WEST COAST</td>
      <td>2.327</td>
      <td>93.063</td>
      <td>20.0</td>
      <td>8.6</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.set_index("Year", inplace=True)
```


```python
df.loc[2000]
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
      <th>Mo</th>
      <th>Dy</th>
      <th>Location Name</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Focal Depth (km)</th>
      <th>Mag</th>
      <th>Total Deaths</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000</th>
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
      <th>2000</th>
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
      <th>2000</th>
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
      <th>2000</th>
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
      <th>2000</th>
      <td>2</td>
      <td>7</td>
      <td>SOUTH AFRICA; SWAZILAND:  MBABANE-MANZINI</td>
      <td>-26.288</td>
      <td>30.888</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>3</td>
      <td>28</td>
      <td>JAPAN:  VOLCANO ISLANDS</td>
      <td>22.338</td>
      <td>143.730</td>
      <td>127.0</td>
      <td>7.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>4</td>
      <td>5</td>
      <td>GREECE:  CRETE</td>
      <td>34.220</td>
      <td>25.690</td>
      <td>38.0</td>
      <td>5.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>5</td>
      <td>4</td>
      <td>INDONESIA:  SULAWESI:  LUWUK, BANGGAI, PELENG,</td>
      <td>-1.105</td>
      <td>123.573</td>
      <td>26.0</td>
      <td>7.6</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>5</td>
      <td>7</td>
      <td>TURKEY:  DOGANYOL, PUTURGE</td>
      <td>38.164</td>
      <td>38.777</td>
      <td>5.0</td>
      <td>4.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>5</td>
      <td>17</td>
      <td>TAIWAN:  TAI-CHUNG COUNTY</td>
      <td>24.223</td>
      <td>121.058</td>
      <td>10.0</td>
      <td>5.4</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6</td>
      <td>4</td>
      <td>INDONESIA:  SUMATRA:  BENGKULU, ENGGANO</td>
      <td>-4.721</td>
      <td>102.087</td>
      <td>33.0</td>
      <td>7.9</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6</td>
      <td>6</td>
      <td>TURKEY:  CERKES, CUBUK, ORTA</td>
      <td>40.693</td>
      <td>32.992</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6</td>
      <td>7</td>
      <td>CHINA:  YUNNAN PROVINCE:  LIUKU; MYANMAR</td>
      <td>26.856</td>
      <td>97.238</td>
      <td>33.0</td>
      <td>6.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6</td>
      <td>7</td>
      <td>INDONESIA:  SOUTHERN SUMATERA:  LAHAT</td>
      <td>-4.612</td>
      <td>101.905</td>
      <td>33.0</td>
      <td>6.7</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6</td>
      <td>10</td>
      <td>TAIWAN:  NAN-TOU</td>
      <td>23.843</td>
      <td>121.225</td>
      <td>33.0</td>
      <td>6.4</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6</td>
      <td>17</td>
      <td>ICELAND:  VESTMANNAEYJAR, HELLA</td>
      <td>63.966</td>
      <td>-20.487</td>
      <td>10.0</td>
      <td>6.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6</td>
      <td>18</td>
      <td>AUSTRALIA:  S, COCOS ISLANDS</td>
      <td>-13.802</td>
      <td>97.453</td>
      <td>10.0</td>
      <td>7.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6</td>
      <td>21</td>
      <td>ICELAND:  GRIMSNES, SELFOSS, EYRARBAKKI, STOKK...</td>
      <td>63.980</td>
      <td>-20.758</td>
      <td>10.0</td>
      <td>6.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>7</td>
      <td>1</td>
      <td>JAPAN:  NEAR S COAST HONSHU:  KOZU-SHIMA</td>
      <td>34.221</td>
      <td>139.131</td>
      <td>10.0</td>
      <td>6.1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>7</td>
      <td>6</td>
      <td>NICARAGUA:  MASAYA</td>
      <td>11.884</td>
      <td>-85.988</td>
      <td>33.0</td>
      <td>5.4</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>7</td>
      <td>12</td>
      <td>INDONESIA: JAWA:BANDUNG,CIBADAK,CIMANDIRI,KADU...</td>
      <td>-6.675</td>
      <td>106.845</td>
      <td>33.0</td>
      <td>5.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>7</td>
      <td>15</td>
      <td>JAPAN:  NEAR S COAST HONSHU:  NII-JIMA</td>
      <td>34.319</td>
      <td>139.260</td>
      <td>10.0</td>
      <td>6.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>7</td>
      <td>16</td>
      <td>PHILIPPINES:  BASCO, MOUNT IRADA, BATAN ISLANDS</td>
      <td>20.253</td>
      <td>122.043</td>
      <td>33.0</td>
      <td>6.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>7</td>
      <td>30</td>
      <td>JAPAN:  HONSHU:  S</td>
      <td>33.901</td>
      <td>139.376</td>
      <td>10.0</td>
      <td>6.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>8</td>
      <td>4</td>
      <td>RUSSIA:  SAKHALIN ISLAND, UGLEGORSK, MAKAROV</td>
      <td>48.786</td>
      <td>142.246</td>
      <td>10.0</td>
      <td>6.8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>8</td>
      <td>21</td>
      <td>CHINA:  YUNNAN PROVINCE:  WUDING</td>
      <td>25.826</td>
      <td>102.194</td>
      <td>33.0</td>
      <td>4.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>9</td>
      <td>3</td>
      <td>CALIFORNIA:  NAPA</td>
      <td>38.379</td>
      <td>-122.413</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>10</td>
      <td>2</td>
      <td>TANZANIA:  NKANSI, RUKWA</td>
      <td>-7.977</td>
      <td>30.709</td>
      <td>34.0</td>
      <td>6.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>10</td>
      <td>6</td>
      <td>JAPAN:  HONSHU:  W:  OKAYAMA, TOTTORI</td>
      <td>35.456</td>
      <td>133.134</td>
      <td>10.0</td>
      <td>6.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>10</td>
      <td>30</td>
      <td>AFGHANISTAN-TAJIKISTAN:  RAKHOR</td>
      <td>37.542</td>
      <td>69.582</td>
      <td>33.0</td>
      <td>5.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>11</td>
      <td>8</td>
      <td>PANAMA-COLOMBIA:  JURADO</td>
      <td>7.042</td>
      <td>-77.829</td>
      <td>17.0</td>
      <td>6.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>11</td>
      <td>16</td>
      <td>PAPUA NEW GUINEA:  NEW IRELAND, DUKE OF YORK</td>
      <td>-4.001</td>
      <td>152.327</td>
      <td>17.0</td>
      <td>8.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>11</td>
      <td>16</td>
      <td>PAPUA NEW GUINEA:  NEW IRELAND, NEW BRITAIN</td>
      <td>-5.233</td>
      <td>153.102</td>
      <td>30.0</td>
      <td>7.8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>11</td>
      <td>17</td>
      <td>PAPUA NEW GUINEA:  NEW BRITAIN</td>
      <td>-5.496</td>
      <td>151.781</td>
      <td>33.0</td>
      <td>7.8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>11</td>
      <td>25</td>
      <td>AZERBAIJAN:  BAKU</td>
      <td>40.245</td>
      <td>49.946</td>
      <td>50.0</td>
      <td>6.8</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>12</td>
      <td>6</td>
      <td>TURKMENISTAN:  NEBITDAG-TURKMENBASHI</td>
      <td>39.566</td>
      <td>54.799</td>
      <td>30.0</td>
      <td>7.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>12</td>
      <td>15</td>
      <td>TURKEY:  AFYON-BOLVADIN</td>
      <td>38.457</td>
      <td>31.351</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



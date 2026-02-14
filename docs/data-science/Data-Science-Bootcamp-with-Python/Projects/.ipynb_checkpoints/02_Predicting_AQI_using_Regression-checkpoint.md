## Importing Required Libraries


```python
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## Reading AQI Data CSV File using Pandas


```python
csv_path = "D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\AQI_data.csv"
df = pd.read_csv(csv_path)
```


```python
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
      <th>City</th>
      <th>Date</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>O3</th>
      <th>Benzene</th>
      <th>Toluene</th>
      <th>Xylene</th>
      <th>AQI</th>
      <th>AQI_Bucket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ahmedabad</td>
      <td>01-01-2015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.92</td>
      <td>18.22</td>
      <td>17.15</td>
      <td>NaN</td>
      <td>0.92</td>
      <td>27.64</td>
      <td>133.36</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ahmedabad</td>
      <td>02-01-2015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.97</td>
      <td>15.69</td>
      <td>16.46</td>
      <td>NaN</td>
      <td>0.97</td>
      <td>24.55</td>
      <td>34.06</td>
      <td>3.68</td>
      <td>5.50</td>
      <td>3.77</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ahmedabad</td>
      <td>03-01-2015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.40</td>
      <td>19.30</td>
      <td>29.70</td>
      <td>NaN</td>
      <td>17.40</td>
      <td>29.07</td>
      <td>30.70</td>
      <td>6.80</td>
      <td>16.40</td>
      <td>2.25</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ahmedabad</td>
      <td>04-01-2015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.70</td>
      <td>18.48</td>
      <td>17.97</td>
      <td>NaN</td>
      <td>1.70</td>
      <td>18.59</td>
      <td>36.08</td>
      <td>4.43</td>
      <td>10.14</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ahmedabad</td>
      <td>05-01-2015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22.10</td>
      <td>21.42</td>
      <td>37.76</td>
      <td>NaN</td>
      <td>22.10</td>
      <td>39.33</td>
      <td>39.31</td>
      <td>7.01</td>
      <td>18.89</td>
      <td>2.78</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (29531, 16)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 29531 entries, 0 to 29530
    Data columns (total 16 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   City        29531 non-null  object 
     1   Date        29531 non-null  object 
     2   PM2.5       24933 non-null  float64
     3   PM10        18391 non-null  float64
     4   NO          25949 non-null  float64
     5   NO2         25946 non-null  float64
     6   NOx         25346 non-null  float64
     7   NH3         19203 non-null  float64
     8   CO          27472 non-null  float64
     9   SO2         25677 non-null  float64
     10  O3          25509 non-null  float64
     11  Benzene     23908 non-null  float64
     12  Toluene     21490 non-null  float64
     13  Xylene      11422 non-null  float64
     14  AQI         24850 non-null  float64
     15  AQI_Bucket  24850 non-null  object 
    dtypes: float64(13), object(3)
    memory usage: 3.6+ MB
    


```python
df.isnull().sum()
```




    City              0
    Date              0
    PM2.5          4598
    PM10          11140
    NO             3582
    NO2            3585
    NOx            4185
    NH3           10328
    CO             2059
    SO2            3854
    O3             4022
    Benzene        5623
    Toluene        8041
    Xylene        18109
    AQI            4681
    AQI_Bucket     4681
    dtype: int64




```python
df.describe()
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
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>O3</th>
      <th>Benzene</th>
      <th>Toluene</th>
      <th>Xylene</th>
      <th>AQI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>24933.000000</td>
      <td>18391.000000</td>
      <td>25949.000000</td>
      <td>25946.000000</td>
      <td>25346.000000</td>
      <td>19203.000000</td>
      <td>27472.000000</td>
      <td>25677.000000</td>
      <td>25509.000000</td>
      <td>23908.000000</td>
      <td>21490.000000</td>
      <td>11422.000000</td>
      <td>24850.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>67.450578</td>
      <td>118.127103</td>
      <td>17.574730</td>
      <td>28.560659</td>
      <td>32.309123</td>
      <td>23.483476</td>
      <td>2.248598</td>
      <td>14.531977</td>
      <td>34.491430</td>
      <td>3.280840</td>
      <td>8.700972</td>
      <td>3.070128</td>
      <td>166.463581</td>
    </tr>
    <tr>
      <th>std</th>
      <td>64.661449</td>
      <td>90.605110</td>
      <td>22.785846</td>
      <td>24.474746</td>
      <td>31.646011</td>
      <td>25.684275</td>
      <td>6.962884</td>
      <td>18.133775</td>
      <td>21.694928</td>
      <td>15.811136</td>
      <td>19.969164</td>
      <td>6.323247</td>
      <td>140.696585</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.040000</td>
      <td>0.010000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.820000</td>
      <td>56.255000</td>
      <td>5.630000</td>
      <td>11.750000</td>
      <td>12.820000</td>
      <td>8.580000</td>
      <td>0.510000</td>
      <td>5.670000</td>
      <td>18.860000</td>
      <td>0.120000</td>
      <td>0.600000</td>
      <td>0.140000</td>
      <td>81.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>48.570000</td>
      <td>95.680000</td>
      <td>9.890000</td>
      <td>21.690000</td>
      <td>23.520000</td>
      <td>15.850000</td>
      <td>0.890000</td>
      <td>9.160000</td>
      <td>30.840000</td>
      <td>1.070000</td>
      <td>2.970000</td>
      <td>0.980000</td>
      <td>118.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>80.590000</td>
      <td>149.745000</td>
      <td>19.950000</td>
      <td>37.620000</td>
      <td>40.127500</td>
      <td>30.020000</td>
      <td>1.450000</td>
      <td>15.220000</td>
      <td>45.570000</td>
      <td>3.080000</td>
      <td>9.150000</td>
      <td>3.350000</td>
      <td>208.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>949.990000</td>
      <td>1000.000000</td>
      <td>390.680000</td>
      <td>362.210000</td>
      <td>467.630000</td>
      <td>352.890000</td>
      <td>175.810000</td>
      <td>193.860000</td>
      <td>257.730000</td>
      <td>455.030000</td>
      <td>454.850000</td>
      <td>170.370000</td>
      <td>2049.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.nunique()
```




    City             26
    Date           2009
    PM2.5         11716
    PM10          12571
    NO             5776
    NO2            7404
    NOx            8156
    NH3            5922
    CO             1779
    SO2            4761
    O3             7699
    Benzene        1873
    Toluene        3608
    Xylene         1561
    AQI             829
    AQI_Bucket        6
    dtype: int64




```python
df.columns
```




    Index(['City', 'Date', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
           'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI', 'AQI_Bucket'],
          dtype='object')




```python
sns.barplot(x="City", y="AQI", data=df)

```




    <Axes: xlabel='City', ylabel='AQI'>




    
![png](02_Predicting_AQI_using_Regression-checkpoint_files/02_Predicting_AQI_using_Regression-checkpoint_11_1.png)
    


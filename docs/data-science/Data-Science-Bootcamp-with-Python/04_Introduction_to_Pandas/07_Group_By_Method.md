[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/04_Introduction_to_Pandas/07_Group_By_Method.ipynb)

# Group By (Split, Apply, Combine)
One of the most powerful features of Pandas is its ability to group data and perform aggregations on the grouped data. The groupby() method is used to group data in Pandas based on one or more columns.

Split-Apply-Combine is a common pattern in data analysis where the data is first split into groups based on one or more criteria, then a function is applied to each group individually, and finally the results are combined into a single data structure.

Here's a breakdown of each step in the Split-Apply-Combine process:

* Split: The data is split into smaller groups based on one or more criteria. For example, we might split data based on the values in a particular column or based on a time period.

* Apply: A function is applied to each group individually. This function could be an aggregation function like sum() or mean(), or it could be a transformation function that modifies the data within each group.

* Combine: The results from each group are combined back together into a single data structure. This could be a new DataFrame or Series, or it could be a summary statistic like a mean or a standard deviation.


```python
import pandas as pd
```


```python
# Creating a dataframe
filepath = r"D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\04_Introduction_to_Pandas\Datasets\weather_by_cities.csv"
df1 = pd.read_csv(filepath)
df1
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
      <th>day</th>
      <th>city</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01-01-2017</td>
      <td>new york</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2017</td>
      <td>new york</td>
      <td>36</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2017</td>
      <td>new york</td>
      <td>28</td>
      <td>12</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2017</td>
      <td>new york</td>
      <td>33</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-01-2017</td>
      <td>mumbai</td>
      <td>90</td>
      <td>5</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-02-2017</td>
      <td>mumbai</td>
      <td>85</td>
      <td>12</td>
      <td>Fog</td>
    </tr>
    <tr>
      <th>6</th>
      <td>01-03-2017</td>
      <td>mumbai</td>
      <td>87</td>
      <td>15</td>
      <td>Fog</td>
    </tr>
    <tr>
      <th>7</th>
      <td>01-04-2017</td>
      <td>mumbai</td>
      <td>92</td>
      <td>5</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>8</th>
      <td>01-01-2017</td>
      <td>paris</td>
      <td>45</td>
      <td>20</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>9</th>
      <td>01-02-2017</td>
      <td>paris</td>
      <td>50</td>
      <td>13</td>
      <td>Cloudy</td>
    </tr>
    <tr>
      <th>10</th>
      <td>01-03-2017</td>
      <td>paris</td>
      <td>54</td>
      <td>8</td>
      <td>Cloudy</td>
    </tr>
    <tr>
      <th>11</th>
      <td>01-04-2017</td>
      <td>paris</td>
      <td>42</td>
      <td>10</td>
      <td>Cloudy</td>
    </tr>
  </tbody>
</table>
</div>



## 01. Use groupby() Method
The groupby() method is used to group data in Pandas. It takes one or more column names as arguments and returns a GroupBy object.


```python
groups = df1.groupby("city")
groups
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000021627575EE0>



## 02. groupby() Representation Internally


```python
for city, city_df in groups:
    print(city)
    print(city_df)
```

    mumbai
              day    city  temperature  windspeed  event
    4  01-01-2017  mumbai           90          5  Sunny
    5  01-02-2017  mumbai           85         12    Fog
    6  01-03-2017  mumbai           87         15    Fog
    7  01-04-2017  mumbai           92          5   Rain
    new york
              day      city  temperature  windspeed  event
    0  01-01-2017  new york           32          6   Rain
    1  01-02-2017  new york           36          7  Sunny
    2  01-03-2017  new york           28         12   Snow
    3  01-04-2017  new york           33          7  Sunny
    paris
               day   city  temperature  windspeed   event
    8   01-01-2017  paris           45         20   Sunny
    9   01-02-2017  paris           50         13  Cloudy
    10  01-03-2017  paris           54          8  Cloudy
    11  01-04-2017  paris           42         10  Cloudy


## 03. Aggregating Data using groupby()
Once the data is grouped, we can perform various aggregation functions on the data. Some of the commonly used aggregation functions are:

* sum(): returns the sum of values in each group
* mean(): returns the mean of values in each group
* median(): returns the median of values in each group
* min(): returns the minimum value in each group
* max(): returns the maximum value in each group
* count(): returns the number of values in each group


```python
groups.max()
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mumbai</th>
      <td>01-04-2017</td>
      <td>92</td>
      <td>15</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>new york</th>
      <td>01-04-2017</td>
      <td>36</td>
      <td>12</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>paris</th>
      <td>01-04-2017</td>
      <td>54</td>
      <td>20</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
groups.mean()
```

    C:\Users\KRISH\AppData\Local\Temp\ipykernel_3440\642604181.py:1: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      groups.mean()
    




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
      <th>temperature</th>
      <th>windspeed</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mumbai</th>
      <td>88.50</td>
      <td>9.25</td>
    </tr>
    <tr>
      <th>new york</th>
      <td>32.25</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>paris</th>
      <td>47.75</td>
      <td>12.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
groups.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">temperature</th>
      <th colspan="8" halign="left">windspeed</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>mumbai</th>
      <td>4.0</td>
      <td>88.50</td>
      <td>3.109126</td>
      <td>85.0</td>
      <td>86.50</td>
      <td>88.5</td>
      <td>90.50</td>
      <td>92.0</td>
      <td>4.0</td>
      <td>9.25</td>
      <td>5.057997</td>
      <td>5.0</td>
      <td>5.00</td>
      <td>8.5</td>
      <td>12.75</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>new york</th>
      <td>4.0</td>
      <td>32.25</td>
      <td>3.304038</td>
      <td>28.0</td>
      <td>31.00</td>
      <td>32.5</td>
      <td>33.75</td>
      <td>36.0</td>
      <td>4.0</td>
      <td>8.00</td>
      <td>2.708013</td>
      <td>6.0</td>
      <td>6.75</td>
      <td>7.0</td>
      <td>8.25</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>paris</th>
      <td>4.0</td>
      <td>47.75</td>
      <td>5.315073</td>
      <td>42.0</td>
      <td>44.25</td>
      <td>47.5</td>
      <td>51.00</td>
      <td>54.0</td>
      <td>4.0</td>
      <td>12.75</td>
      <td>5.251984</td>
      <td>8.0</td>
      <td>9.50</td>
      <td>11.5</td>
      <td>14.75</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>



## 04. Plotting groupby() Data


```python
%matplotlib inline
groups.plot()
```




    city
    mumbai      Axes(0.125,0.11;0.775x0.77)
    new york    Axes(0.125,0.11;0.775x0.77)
    paris       Axes(0.125,0.11;0.775x0.77)
    dtype: object




    
![png](07_Group_By_Method_files/07_Group_By_Method_12_1.png)
    




![png](07_Group_By_Method_files/07_Group_By_Method_12_2.png)
    




![png](07_Group_By_Method_files/07_Group_By_Method_12_3.png)
    


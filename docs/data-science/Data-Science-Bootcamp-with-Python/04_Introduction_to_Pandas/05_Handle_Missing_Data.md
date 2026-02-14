[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/04_Introduction_to_Pandas/05_Handle_Missing_Data.ipynb)

# Handle Missing Data
Handling missing data is an important part of data analysis, and Pandas provides a number of methods for dealing with missing values. In this notebook, we will cover some common techniques for handling missing data using Pandas.


```python
import pandas as pd
```


```python
# Creating a dataframe from CSV
df1 = pd.read_csv("weather_data.csv")
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01-01-2020</td>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-04-2020</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-05-2020</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-06-2020</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-07-2020</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-08-2020</td>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 01. Convert String Column into Date Type
In Pandas, it is common to work with data that includes dates. However, sometimes the dates are stored as strings, which makes it difficult to perform any operations on them. In this case, it is necessary to convert the string column into a date type.

Pandas provides the to_datetime() method for converting a string column into a date type. This method is very powerful and flexible, allowing you to convert many different string formats into dates.


```python
# Print the datatype of values in 'day' column
type(df1.day[0])
```




    str




```python
df2 = pd.read_csv("weather_data.csv", parse_dates=["day"])
df2
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-01</td>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-04</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-06</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-07</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-01-08</td>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the datatype of values in 'day' column
type(df2.day[0])
```




    pandas._libs.tslibs.timestamps.Timestamp



## 02. Use Date as Index of DataFrame
To use a date column as the index of a DataFrame, we can use the set_index() method of the DataFrame object, and pass the name of the date column as an argument. The set_index() method will return a new DataFrame with the specified column as the index.

The inplace=True argument is used to modify the DataFrame in place, rather than creating a new one.


```python
df3 = pd.read_csv("weather_data.csv", parse_dates=["day"])
df3.set_index("day", inplace=True)
df3
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 03. Use fillna() Method
In Pandas, fillna() is a method used to fill missing or null values in a DataFrame with a specified value or technique. This method can be used to clean up the data before further processing.


```python
df4 = df3.fillna(0)
df4
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>0.0</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>0.0</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>0.0</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
df5 = df3.fillna({
    "temperature": 0,
    "windspeed": 0,
    "event": "no event"
})
df5
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>0.0</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>no event</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>0.0</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>0.0</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 04. Use fillna(method="ffill"/"bfill") Method
The fillna() method in pandas is used to fill the missing or NaN values in a DataFrame with a specified value or method. The method parameter can be used to fill the missing values using forward or backward filling method. When method='ffill', it fills the missing values with the previous non-missing value along each column. When method='bfill', it fills the missing values with the next non-missing value along each column.


```python
# Using fillna(method="ffill") Method
df6 = df3.fillna(method="ffill")
df6
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>32.0</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>32.0</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>7.0</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>24.0</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>4.0</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Using fillna(method="bfill") Method
df7 = df3.fillna(method="bfill")
df7
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>24.0</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>24.0</td>
      <td>4.0</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>4.0</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>32.0</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



### 01. 'axis' Parameter in fillna() Method
The fillna() method in pandas is used to fill the missing or NaN values in a DataFrame with a specified value or method. The axis parameter is used to specify the direction in which the fill operation should be applied.

The axis parameter can be set to 0 or 'index' to apply the fill operation along the rows, and to 1 or 'columns' to apply the fill operation along the columns.


```python
df8 = df3.fillna(method="ffill", axis="columns")
df8
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>24.0</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>32.0</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
df9 = df3.fillna(method="bfill", axis=1)
df9
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>7.0</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>Snow</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>Sunny</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



### 02. 'limit' Parameter in fillna() Method
The limit parameter in the fillna() method is used to specify the maximum number of consecutive NaN values to be filled. This parameter takes an integer value.


```python
df10 = df3.fillna(method="ffill", limit=1)
df10
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>32.0</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>24.0</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>4.0</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 05. Use interpolate() Method to do Interpolation
The interpolate() method in Pandas is used to fill the missing values (NaN) in a DataFrame or a Series by using various interpolation techniques. The interpolate() method supports several interpolation techniques, including linear, quadratic, cubic, and more. You can specify the interpolation technique by passing a value for the method parameter. For example, to use quadratic interpolation, you can set method='quadratic'


```python
df3
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
df11 = df3.interpolate()
df11
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.000000</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>29.333333</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>26.666667</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.000000</td>
      <td>5.0</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>28.000000</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.000000</td>
      <td>4.0</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



### 01. interpolate() Method 'time'
The interpolate() method in Pandas can perform different types of interpolation, including linear, polynomial, and time-based interpolation. Time-based interpolation is used when dealing with time series data, where missing values are filled based on the time difference between observations.

To perform time-based interpolation, the interpolate() method needs to be called with the method parameter set to 'time'. This method uses the time stamps of the available data points to estimate the values of missing data points.


```python
df12 = df3.interpolate(method="time")
df12
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>27.2</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>25.6</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>5.0</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>28.0</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>4.0</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 06. Use dropna() Method to Drop all the rows With 'NaN' Values
In Pandas, the dropna() method is used to remove the rows or columns containing NaN or missing data. This method is useful when we have missing values in our dataset and want to remove them.


```python
df13 = pd.read_csv("weather_data.csv", parse_dates=["day"])
df13.set_index("day", inplace=True)
df13
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dropping all the rows With 'NaN' Values
df14 = df13.dropna()
df14
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
  </tbody>
</table>
</div>



### 01. 'how' Parameter in dropna() Method 
The how parameter in the dropna() method of pandas is used to determine the condition for dropping the rows or columns containing the missing values (NaN values).
* how='any': This is the default option. If any missing value is present in a row or column, the entire row or column will be dropped.
* how='all': Only the rows or columns containing all missing values will be dropped.
* how='thresh': Only the rows or columns containing a minimum number of non-missing values, specified by the thresh parameter, will be kept.


```python
# Using dropna(how="all")
df15 = df13.dropna(how="all")
df15
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



### 02. 'thresh' Parameter in dropna() Method
The thresh parameter in the dropna() method of pandas is used in combination with the how='thresh' option to specify the minimum number of non-missing values required to keep a row or column.

For example, if thresh=2, only the rows or columns containing at least two non-missing values will be kept, and all others will be dropped.


```python
df13
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
df16 = df13.dropna(thresh=2)
df16
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 07. Inserting Missing Dates
In Pandas, missing dates can be inserted into a DataFrame using the reindex() method. This method returns a new DataFrame with the specified index.

To insert missing dates, we first need to create a range of dates that includes all the missing dates we want to add. We can do this using the pd.date_range() function. This function takes a start and end date, and returns a range of dates in between.


```python
df17 = pd.read_csv("weather_data.csv", parse_dates=["day"])
df17.set_index("day", inplace=True)
df17
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Inserting missing dates
date_range = pd.date_range("01-01-2020", "01-08-2020")
date_id = pd.DatetimeIndex(date_range)
df18 = df17.reindex(date_id)
df18
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>NaN</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>NaN</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filling missing values
# Filling numeric values using interpolate method
df18.interpolate(inplace=True)
# Filling non-numeric values using fillna method
df18.fillna(method="ffill", inplace=True)
df18
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
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>32.0</td>
      <td>6.000000</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>30.4</td>
      <td>6.333333</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>28.8</td>
      <td>6.666667</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>27.2</td>
      <td>7.000000</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-05</th>
      <td>25.6</td>
      <td>6.000000</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>24.0</td>
      <td>5.000000</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>28.0</td>
      <td>4.000000</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>32.0</td>
      <td>4.000000</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



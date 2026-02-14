#  Pandas Dataframe
Pandas DataFrame is a two-dimensional labeled data structure, where the columns can have different data types, such as integers, floats, and strings. It is one of the most popular data structures used in data analysis and machine learning tasks.

In this course, we will cover the basics of using Pandas DataFrame for data analysis. We will start with an introduction to Pandas DataFrame and then move on to topics such as data manipulation, data cleaning, and data visualization.


```python
import pandas as pd
```

## 01. Creating a Pandas DataFrame

### 01. Creating DataFrame from Dictionary


```python
weather_dict = {
    "day": ["1/1/2020", "1/2/2020", "1/3/2020", "1/4/2020", "1/5/2020", "1/6/2020"],
    "temperature": [32, 35, 28, 24, 32, 31],
    "windspeed": [6, 7, 2, 7, 4, 2],
    "event": ["Rain", "Sunny", "Snow", "Snow", "Rain", "Sunny"]
}
```


```python
# Creating dataframe from dictionary
df1 = pd.DataFrame(weather_dict)
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
      <td>1/1/2020</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2020</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2020</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/4/2020</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2020</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1/6/2020</td>
      <td>31</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



### 02. Creating DataFrame from CSV


```python
path = "D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\sample_weather_data.csv"
df2 = pd.read_csv(path)
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
      <td>01-01-2020</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2020</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2020</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2020</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2020</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2020</td>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the type of the df2 variable
type(df2)
```




    pandas.core.frame.DataFrame




```python
# Printing the shape of the dataframe
rows, colums = df2.shape
rows
```




    6



## 02. head() Method
In Pandas, the head() method is used to view the first few rows of a DataFrame. By default, it displays the first 5 rows of the DataFrame. This method is useful to get a quick overview of the data in the DataFrame.


```python
# Print first five rows of the dataframe
df2.head()
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
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2020</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2020</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2020</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2020</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the first two rows of the dataframe
df2.head(2)
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
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2020</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 03. tail() Method
In Pandas, the tail() method is used to view the last few rows of a DataFrame. By default, it displays the last 5 rows of the DataFrame. This method is useful to get a quick overview of the data in the DataFrame.


```python
# Print last five rows of dataframe
df2.tail()
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
      <th>1</th>
      <td>01-02-2020</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2020</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2020</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2020</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2020</td>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the last two rows of the dataframe
df2.tail(2)
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
      <th>4</th>
      <td>01-05-2020</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2020</td>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 04. Indexing and Slicing in DataFrame


```python
# Print row number 2 to 4
df2[2:5]
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
      <th>2</th>
      <td>01-03-2020</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2020</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2020</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the names of the columns
df2.columns
```




    Index(['day', 'temperature', 'windspeed', 'event'], dtype='object')




```python
# Print individual column of the dataframe
df2.day
```




    0    01-01-2020
    1    01-02-2020
    2    01-03-2020
    3    01-04-2020
    4    01-05-2020
    5    01-06-2020
    Name: day, dtype: object




```python
# Print the type of a column
type(df2["event"])
```




    pandas.core.series.Series




```python
# Print specific columns from dataframe
df2[["day", "temperature", "event"]]
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
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01-01-2020</td>
      <td>32</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2020</td>
      <td>35</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2020</td>
      <td>28</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2020</td>
      <td>24</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2020</td>
      <td>32</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2020</td>
      <td>32</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 05. Operations with DataFrame

### 01. max() Method


```python
# Print the maximum temperature
df2["temperature"].max()
```




    35



### 02. min() Method


```python
# Print the minimum temperature
df2["temperature"].min()
```




    24



### 03. mean() Method


```python
# Print the mean (average) of the temperature
df2["temperature"].mean()
```




    30.5



### 04. std() Method


```python
# Print the standard deviation of the temperature
df2["temperature"].std()
```




    3.8858718455450894



### 05. describe() Method


```python
# Print the statistics of the whole dataframe
df2.describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>30.500000</td>
      <td>4.666667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.885872</td>
      <td>2.338090</td>
    </tr>
    <tr>
      <th>min</th>
      <td>24.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>29.000000</td>
      <td>2.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>32.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>32.000000</td>
      <td>6.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>35.000000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 06. Conditional Selection in DataFrame


```python
# Print all the rows where temperature greater than or equal to 30
df2[df2.temperature >= 30]
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
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2020</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2020</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2020</td>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the row where temperature is maximum
df2[df2.temperature == df2["temperature"].max()]
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
      <th>1</th>
      <td>01-02-2020</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print only the day and temperature column where the temperature is maximum
df2[["day", "temperature"]][df2.temperature == df2["temperature"].max()]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>01-02-2020</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>



## 07. set_index() Method
In Pandas, the set_index() method is used to set one or more columns as the index of a DataFrame. This method returns a new DataFrame with the specified column(s) set as the index.


```python
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
      <td>01-01-2020</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2020</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2020</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2020</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2020</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2020</td>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.index
```




    RangeIndex(start=0, stop=6, step=1)




```python
# Set the 'day' column as the index of the dataframe
df2.set_index("day", inplace=True)
```


```python
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
      <th>01-01-2020</th>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>01-02-2020</th>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>01-03-2020</th>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>01-04-2020</th>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>01-05-2020</th>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>01-06-2020</th>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The 'loc' function is used to access a group of rows and columns by label(s) or a boolean array.
df2.loc["01-01-2020"]
```




    temperature      32
    windspeed         6
    event          Rain
    Name: 01-01-2020, dtype: object



## 08. reset_index() Method
In Pandas, the reset_index() method is used to reset the index of a DataFrame to a default numbered index. It is often used to reset the index after setting it to a column or multiple columns using the set_index() method.


```python
df2.reset_index(inplace=True)
```


```python
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
      <td>01-01-2020</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2020</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2020</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2020</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2020</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2020</td>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



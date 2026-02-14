[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/04_Introduction_to_Pandas/03_Different_Ways_of_Creating_Dataframe.ipynb)

# Different Ways of Creating DataFrame
Pandas is a powerful library for data manipulation and analysis in Python. It provides the DataFrame object, which is a two-dimensional table-like data structure with rows and columns. In this tutorial, we will cover different ways of creating a DataFrame in Pandas.


```python
import pandas as pd
```

## 01. Creating DataFrame Using read_csv() Method
Pandas provides a read_csv() method which allows us to create a DataFrame by reading a CSV file. This is one of the most common ways of creating a DataFrame in Pandas, especially when working with large datasets.


```python
csv_path = "D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\sample_weather_data.csv"
df1 = pd.read_csv(csv_path)
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



## 02. Creating DataFrame Using read_excel() Method
In addition to read_csv(), Pandas also provides a read_excel() method which allows us to create a DataFrame by reading an Excel file.


```python
xls_path = "D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\sample_weather_data.xlsx"
df2 = pd.read_excel(xls_path)
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
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-02-01</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-03-01</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-01</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-05-01</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-06-01</td>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 03. Creating DataFrame from Dictionary
Another way to create a DataFrame in Pandas is by using a Python dictionary. The keys of the dictionary represent the column names of the DataFrame, while the values represent the data for each column. The values can be of any data type that can be represented in a Pandas DataFrame (such as lists, NumPy arrays, or Pandas Series).


```python
weather_dict = {
    "day": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"],
    "temperature": [32, 35, 28, 24, 32, 32],
    "windspeed": [6, 7, 2, 7, 4, 2],
    "event": ["Rain", "Sunny", "Snow", "Snow", "Rain", "Sunny"]
}
```


```python
df3 = pd.DataFrame(weather_dict)
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
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-02-01</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-03-01</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-01</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-05-01</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-06-01</td>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 04. Creating DataFrame from a List of Tuples
Another way to create a DataFrame in Pandas is by using a list of tuples. In this case, you need to provide the column names. The column names are optional, but if you don't specify them, Pandas will assign default column names (0, 1, 2, etc.) to the DataFrame.


```python
weather_data = [
	("2020-01-01", 32, 6, "Rain"),
	("2020-02-01", 35, 7, "Sunny"),
	("2020-03-01", 28, 2, "Snow"),
	("2020-04-01", 24, 7, "Snow"),
	("2020-05-01", 32, 4, "Rain"),
	("2020-06-01", 32, 2, "Sunny")
]
```


```python
df4 = pd.DataFrame(weather_data, columns=["day", "temperature", "windspeed", "event"])
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
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-02-01</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-03-01</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-01</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-05-01</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-06-01</td>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



## 05. Creating DataFrame Using List of Dictionaries
Another way to create a DataFrame in Pandas is by using a list of dictionaries. In this case, the keys of each dictionary are used as column names in the resulting DataFrame. The order of the keys in the first dictionary determines the order of the columns in the DataFrame.


```python
weather_dict_list = [
    {"day": "2020-01-01", "temperature": 32, "windspeed": 6, "event": "Rain"},
    {"day": "2020-02-01", "temperature": 35, "windspeed": 7, "event": "Sunny"},
    {"day": "2020-03-01", "temperature": 28, "windspeed": 2, "event": "Snow"},
    {"day": "2020-04-01", "temperature": 24, "windspeed": 7, "event": "Snow"},
    {"day": "2020-05-01", "temperature": 32, "windspeed": 4, "event": "Rain"},
    {"day": "2020-06-01", "temperature": 32, "windspeed": 2, "event": "Sunny"},
]
```


```python
df5 = pd.DataFrame(weather_dict_list)
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
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-02-01</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-03-01</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-01</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-05-01</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-06-01</td>
      <td>32</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>



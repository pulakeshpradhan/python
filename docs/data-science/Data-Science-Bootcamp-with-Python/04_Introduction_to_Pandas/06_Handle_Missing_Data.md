[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/04_Introduction_to_Pandas/06_Handle_Missing_Data.ipynb)

# Handle Missing Data
In data analysis, missing data can be a common occurrence, and handling it properly is important to ensure accurate analysis. Pandas is a powerful library in Python that provides several methods for handling missing data. One of the methods to handle missing data is by using the replace() method.

The replace() method in Pandas can be used to replace a specific value with another value, including replacing missing or null values.


```python
import pandas as pd
```


```python
df1 = pd.read_csv("weather_data2.csv")
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
      <td>01-01-2017</td>
      <td>32F</td>
      <td>6 mph</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2017</td>
      <td>-99999</td>
      <td>7 mph</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2017</td>
      <td>28</td>
      <td>-99999</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2017</td>
      <td>-99999</td>
      <td>7</td>
      <td>No Event</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2017</td>
      <td>32</td>
      <td>-88888</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2017</td>
      <td>31</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>6</th>
      <td>01-06-2017</td>
      <td>34</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 01. Use replace() Method to Replace Values
Pandas provides the replace() method to replace values in a DataFrame. The replace() method can be used to replace a single value or multiple values at once.


```python
# Replacing single value
df2 = df1.replace(-99999, "NaN")
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
      <td>01-01-2017</td>
      <td>32F</td>
      <td>6 mph</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2017</td>
      <td>-99999</td>
      <td>7 mph</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2017</td>
      <td>28</td>
      <td>-99999</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2017</td>
      <td>-99999</td>
      <td>7</td>
      <td>No Event</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2017</td>
      <td>32</td>
      <td>-88888</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2017</td>
      <td>31</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>6</th>
      <td>01-06-2017</td>
      <td>34</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Replacing multiple values
df3 = df1.replace([-99999, -88888], "NaN")
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
      <td>01-01-2017</td>
      <td>32F</td>
      <td>6 mph</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2017</td>
      <td>-99999</td>
      <td>7 mph</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2017</td>
      <td>28</td>
      <td>-99999</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2017</td>
      <td>-99999</td>
      <td>7</td>
      <td>No Event</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2017</td>
      <td>32</td>
      <td>-88888</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2017</td>
      <td>31</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>6</th>
      <td>01-06-2017</td>
      <td>34</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Replacing values based on specific columns
df4 = df1.replace({
    "temperature": -99999,
    "windspeed": [-99999, -88888],
    "event": "0"
}, "NaN")
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
      <td>01-01-2017</td>
      <td>32F</td>
      <td>6 mph</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2017</td>
      <td>-99999</td>
      <td>7 mph</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2017</td>
      <td>28</td>
      <td>-99999</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2017</td>
      <td>-99999</td>
      <td>7</td>
      <td>No Event</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2017</td>
      <td>32</td>
      <td>-88888</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2017</td>
      <td>31</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>6</th>
      <td>01-06-2017</td>
      <td>34</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Mapping specific values
df5 = df1.replace({
    -99999: "NaN",
    "No Event": "Sunny"
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01-01-2017</td>
      <td>32F</td>
      <td>6 mph</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2017</td>
      <td>-99999</td>
      <td>7 mph</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2017</td>
      <td>28</td>
      <td>-99999</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2017</td>
      <td>-99999</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2017</td>
      <td>32</td>
      <td>-88888</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2017</td>
      <td>31</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>6</th>
      <td>01-06-2017</td>
      <td>34</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 02. regex() (Regular Expression) Parameter
The regex parameter is a boolean parameter used in Pandas to determine whether to interpret the to_replace parameter in the replace() method as a regular expression. When regex is set to True, to_replace is treated as a regular expression pattern, and any matches are replaced with the value parameter.

Regular expressions are a powerful tool for pattern matching in text data. They allow us to search for patterns in text, and can be used to extract specific parts of a string or replace certain parts of a string with another value.

In the context of Pandas, using regular expressions can be very useful for cleaning and transforming data. For example, we could use regular expressions to extract email addresses from a column of text data or to replace certain characters with others.


```python
# Using regex
df6 = df1.replace("[A-Za-z]", "", regex=True)
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01-01-2017</td>
      <td>32</td>
      <td>6</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2017</td>
      <td>-99999</td>
      <td>7</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2017</td>
      <td>28</td>
      <td>-99999</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2017</td>
      <td>-99999</td>
      <td>7</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2017</td>
      <td>32</td>
      <td>-88888</td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2017</td>
      <td>31</td>
      <td>2</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>01-06-2017</td>
      <td>34</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Using regex based on specific columns
df7 = df1.replace({
    "temperature": "[A-Za-z]",
    "windspeed": "[A-Za-z]"
}, "", regex=True)
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01-01-2017</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-02-2017</td>
      <td>-99999</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-03-2017</td>
      <td>28</td>
      <td>-99999</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-04-2017</td>
      <td>-99999</td>
      <td>7</td>
      <td>No Event</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-05-2017</td>
      <td>32</td>
      <td>-88888</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01-06-2017</td>
      <td>31</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>6</th>
      <td>01-06-2017</td>
      <td>34</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 03. Replace the List of Values with Another List of Values
In Pandas, we can use the replace() method to replace a list of values in a DataFrame column with another list of values. This can be useful when we want to replace multiple values with a single value or when we want to replace values with different values depending on their original value.


```python
# Creatin a new dataframe
df = pd.DataFrame({
    "score": ["exceptional", "average", "good", "poor", "average", "exceptional"],
    "student": ["rob", "maya", "parthiv", "tom", "julian", "erica"]
})
df
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
      <th>score</th>
      <th>student</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>exceptional</td>
      <td>rob</td>
    </tr>
    <tr>
      <th>1</th>
      <td>average</td>
      <td>maya</td>
    </tr>
    <tr>
      <th>2</th>
      <td>good</td>
      <td>parthiv</td>
    </tr>
    <tr>
      <th>3</th>
      <td>poor</td>
      <td>tom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>average</td>
      <td>julian</td>
    </tr>
    <tr>
      <th>5</th>
      <td>exceptional</td>
      <td>erica</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_df = df.replace(["poor", "average", "good", "exceptional"], [1, 2, 3, 4])
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
      <th>score</th>
      <th>student</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>rob</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>maya</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>parthiv</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>tom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>julian</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>erica</td>
    </tr>
  </tbody>
</table>
</div>



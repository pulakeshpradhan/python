[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/04_Introduction_to_Pandas/01_Introduction_to_Pandas.ipynb)

# Introduction to Pandas
Pandas is a Python library that provides powerful data manipulation capabilities. It is built on top of NumPy and provides easy-to-use data structures and data analysis tools for data processing and analysis.

In this module, we will cover the basics of using Pandas for data analysis. We will start with an introduction to the Pandas library and then move on to topics such as data structures, data cleaning, data visualization, and statistical analysis.

**Prerequisites:** Before starting with Pandas, you should have a basic understanding of Python programming and NumPy. If you are new to Python, we recommend taking an introductory Python course before starting with this course.

<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/2560px-Pandas_logo.svg.png" style= "max-width: 350px; height: auto"></center>

## 01. Installation and Importing
To install Pandas, use the pip package manager in the terminal by typing the following command:


```python
# !pip install pandas
```


```python
import pandas as pd
```

## 02. Import a CSV Data


```python
# Creating a dataframe
path = "D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\significant_earthquakes_2000_2020.csv"
df = pd.read_csv(path)
```

## 03. Basic Operations


```python
# Print the first five row of csv
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
# Print the shape of the csv
df.shape
```




    (1206, 9)




```python
# Fill th NaN values with 0
df.fillna(0, inplace=True)
```


```python
# Print the maximum earthquake magnitude
df["Mag"].max()
```




    9.1




```python
# Print the Location Name of the earthquakes where total deaths were greater than 50000
df["Location Name"][df["Total Deaths"] > 50000]
```




    272         INDONESIA:  SUMATRA:  ACEH:  OFF WEST COAST
    320    PAKISTAN:  MUZAFFARABAD, URI, ANANTNAG, BARAMULA
    490                            CHINA:  SICHUAN PROVINCE
    607                              HAITI:  PORT-AU-PRINCE
    Name: Location Name, dtype: object




```python
# Print the Location Name where the magnitude of the earthquake crossed 8.5 
df["Location Name"][df["Mag"] >= 8.5]
```




    272    INDONESIA:  SUMATRA:  ACEH:  OFF WEST COAST
    294                      INDONESIA:  SUMATERA:  SW
    614          CHILE:  MAULE, CONCEPCION, TALCAHUANO
    674                                 JAPAN:  HONSHU
    736         INDONESIA:  N SUMATRA:  OFF WEST COAST
    Name: Location Name, dtype: object




```python
# Print the average focal depth (km) of the earthquakes
df["Focal Depth (km)"].mean()
```




    30.892205638474294



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/00_Data_Gathering/00_Working_with_CSV_Files.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# **Working with CSV Files**


## **Import Required Libraries**

```python
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
```

## **Open a Local csv File**

```python
csv_path = r"D:\Coding\Datasets\Iris.csv"
df = pd.read_csv(csv_path)
df
```

## **Open a csv File from an URL**
1. **requests Module:**<br>
The requests module is used for making HTTP requests to interact with web resources. It simplifies the process of sending HTTP requests and handling the responses. You can use it to make GET, POST, PUT, DELETE, and other types of HTTP requests.

2. **io Module:**<br>
The io module provides classes for working with streams and file-like objects. It's often used to handle input/output operations in a generic way. You can use it to work with in-memory streams, files, and other data sources.


```python
import requests
from io import StringIO
```

```python
url = r"https://media.githubusercontent.com/media/datablist/sample-csv-files/main/files/customers/customers-1000.csv"
req = requests.get(url)
data = StringIO(req.text)
```

```python
pd.read_csv(data)
```

## **'sep' Parameter**
The sep parameter specifies the delimiter or separator used in the CSV file to distinguish between different fields or columns. By default, the comma (,) is used as the separator. However, in some cases, CSV files might use other delimiters such as tabs (\t) or semicolons (;).

```python
# Reading Tab Separated File with pandas
tsv_path = r"D:\Coding\Datasets\movie_titles_metadata.tsv"
pd.read_csv(tsv_path, sep="\t")
```

```python
# Giving the name of columns
column_names = ["sl.no", "name", "release_year", "rating", "votes", "genres"]
pd.read_csv(tsv_path, sep="\t", names=column_names)
```

## **'index_col' Parameter**
The index_col parameter specifies which column(s) should be used as the DataFrame's index. The index is used to uniquely label rows in the DataFrame. By default, pandas assigns a numeric index starting from 0. Specifying index_col allows you to use one or more columns as the index.

```python
pd.read_csv(csv_path, index_col="Id")
```

## **'header' Parameter**
The header parameter specifies which row should be considered as the header (column names) when reading the CSV file. By default, the first row is treated as the header. If you set header=None, pandas will not use any row as the header, and columns will be labeled with numeric indices. You can also provide an integer row number to use as the header, or a list of integers to skip multiple initial rows.

```python
test_csv_path = r"D:\Coding\Datasets\test.csv"
pd.read_csv(test_csv_path)
```

```python
# Changing the header row
pd.read_csv(test_csv_path, header=1)
```

## **'use_cols' Parameter**
The usecols parameter in the pandas read_csv() function is used to specify which columns from the CSV file should be read into the DataFrame. This parameter allows you to selectively read only a subset of columns, which can be useful when dealing with large datasets where not all columns are needed or when you want to focus on specific data.

```python
train_csv_path = r"D:\Coding\Datasets\aug_train.csv"
pd.read_csv(train_csv_path)
```

```python
# Create a list of required columns
required_columns = ["enrollee_id", "gender", "education_level"]
pd.read_csv(train_csv_path, usecols=required_columns)
```

## **'squeeze' Parameters**
The squeeze parameter is used when reading a CSV file that has only one column. It determines whether the resulting DataFrame should be "squeezed" into a Series if the CSV data consists of a single column. This can be particularly useful to simplify your data structure when you're dealing with single-column datasets.

```python
pd.read_csv(train_csv_path, usecols=["gender"], squeeze=True)
```

## **'skiprows'/'nrows' Parameter**
1. **skiprows Parameter:**<br>
The skiprows parameter allows you to specify the number of rows at the beginning of the CSV file to skip while reading. You can pass an integer representing the number of rows to skip or a list of row indices (0-based) that should be skipped.

2. **nrows Parameter:**<br>
The nrows parameter is used to limit the number of rows read from the CSV file. You can pass an integer representing the maximum number of rows to read.

```python
pd.read_csv(train_csv_path)
```

```python
# Skipping the rows at index 1 and  3
pd.read_csv(train_csv_path, skiprows=[1, 3])
```

```python
# Reading only the first 100 rows
pd.read_csv(train_csv_path, nrows=100)
```

## **'encodng' Parameter**
The encoding parameter is used to specify the character encoding of the CSV file being read. Character encoding defines how characters are represented as bytes in a file. Different encodings are used for different languages and writing systems.

When reading a CSV file, it's important to use the correct encoding to ensure that the data is interpreted correctly. If you encounter issues where characters are not displayed correctly or you see encoding-related errors, specifying the appropriate encoding can help resolve these problems.

```python
zomato_csv_path = r"D:\Coding\Datasets\zomato.csv"
# pd.read_csv(zomato_csv_path) # Throws an error
pd.read_csv(zomato_csv_path, encoding="latin-1")
```

## **Skip Bad Lines**
The error_bad_lines parameter is used in the pandas read_csv() function to control how the function handles lines in a CSV file that have too many fields (columns) compared to the expected number of columns.

```python
books_csv_path = r"D:\Coding\Datasets\BX-Books.csv"
pd.read_csv(books_csv_path, encoding="latin-1", error_bad_lines=False)
```

## **'dtype' Parameter**
The dtype parameter allows you to explicitly specify the data types for columns when reading a CSV file into a DataFrame. This can be useful when you want to ensure that specific columns are interpreted with the correct data types.

```python
pd.read_csv(train_csv_path)
```

```python
pd.read_csv(train_csv_path).info()
```

```python
# Converting the datatype of the 'target' column into integer
pd.read_csv(train_csv_path, dtype={"target": "int8"}).info()
```

## **Handling Dates**
The parse_dates parameter allows you to specify columns that should be parsed as datetime objects when reading a CSV file. This can be especially useful when you have date or datetime information in your CSV file that you want to directly interpret as datetime objects in your DataFrame.

```python
ipl_csv_path = r"D:\Coding\Datasets\IPL Matches 2008-2020.csv"
pd.read_csv(ipl_csv_path)
```

```python
pd.read_csv(ipl_csv_path).info()
```

```python
pd.read_csv(ipl_csv_path, parse_dates=["date"]).info()
```

## **'converters' Parameter**
The converters parameter in the pandas read_csv() function is used to provide a dictionary of functions that allow you to customize the way specific columns are converted or transformed during the reading process. It's a powerful tool when you need more control over how the data is processed as it's being read from the CSV file.

```python
pd.read_csv(ipl_csv_path).head()
```

```python
# Creating a function to convert the name of the team into acronym
def renameTeam(name):
    if name == "Kolkata Knight Riders":
        return "KKR"
    else: 
        return name
```

```python
# Applying converters
pd.read_csv(ipl_csv_path, converters={"team2": renameTeam})
```

## **'na_values' Parameters**
The na_values parameter is used to specify a list of values that should be treated as missing or NaN (Not a Number) values when reading a CSV file into a DataFrame. This parameter allows you to define how specific values in your CSV file should be interpreted as missing data.

```python
pd.read_csv(train_csv_path)
```

```python
# Selecting 'Male' value as 'NaN' for example
pd.read_csv(train_csv_path, na_values="Male")
```

## **Load a Huge Dataset in Chunks**
The chunksize parameter in the pandas read_csv() function is used to read a large CSV file in smaller, manageable chunks rather than reading the entire file into memory at once. This can be very useful when dealing with datasets that are too large to fit entirely in memory.

```python
dfs = pd.read_csv(train_csv_path, chunksize=5000)
```

```python
for chunk in dfs:
    print(chunk.shape)
```

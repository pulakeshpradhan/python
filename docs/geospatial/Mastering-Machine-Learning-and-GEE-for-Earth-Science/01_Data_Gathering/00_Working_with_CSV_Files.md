---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/01_Data_Gathering/00_Working_with_CSV_Files.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="qG-L_43NzAdC" -->
# **Working with CSV Files**
Working with CSV (Comma-Separated Values) files in Python using the Pandas library is a common task in data analysis and manipulation. Pandas is a powerful data manipulation library that provides data structures like DataFrame, which is particularly useful for handling tabular data, such as that found in CSV files.
<!-- #endregion -->

<!-- #region id="u4KruUJrzNjS" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8vFe6W8p0RCl" outputId="a8055e59-4d55-48d9-c4bc-26def9d526a8"
# Mount google drive
from google.colab import drive
drive.mount("/content/drive/")
```

```python id="jXantEUmzVPb"
import pandas as pd
```

<!-- #region id="qvhYNv5kzdgq" -->
## **Open a Local CSV File**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="dMADUGvXzkat" outputId="51c509d8-5113-48f6-de42-e60f4117d67f"
# Define the CSV path
csv_path = "/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/Iris.csv"
df = pd.read_csv(csv_path)
df
```

<!-- #region id="2UinvT5Y1HV2" -->
## **Open a CSV File from an URL**

 - **`requests` Module**: <br>
The `requests` module is used for making HTTP requests to interact with web resources. It simplifies the process of sending HTTP requests and handling the responses. You can use it to make GET, POST, PUT, DELETE, and other types of HTTP requests.

- **`io` Module**: <br>
The `io` module provides classes for working with streams and file-like objects. It's often used to handle input/output operations in a generic way. You can use it to work with in-memory streams, files, and other data sources.
<!-- #endregion -->

```python id="c9ZpIe1R1K0Y"
import requests
from io import StringIO
```

```python id="i_YWR-3q2FXF"
url = r"https://raw.githubusercontent.com/codeforamerica/ohana-api/master/data/sample-csv/addresses.csv"
req = requests.get(url)
data = StringIO(req.text)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="VtD_ozK63xD5" outputId="fa71548b-7fb5-4e20-987a-58a632f83720"
df = pd.read_csv(data)
df.head() # by default 'head' function prints the first five rows of the dataframe
```

<!-- #region id="g2kfpif83Ke2" -->
🔑 **Note:** A response with a status code of 200 indicates a successful HTTP request. The HTTP status code 200 `OK` is one of the standard HTTP response codes, and it means that the request was successful, the server processed the request, and the requested information is contained in the response body.
<!-- #endregion -->

<!-- #region id="MTIJck014GEE" -->
## **`sep` Parameter**
The `sep` parameter specifies the delimiter or separator used in the CSV file to distinguish between different fields or columns. By default, the comma (,) is used as the separator. However, in some cases, CSV files might use other delimiters such as tabs (\t) or semicolons (;).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="BFjgCPVr4WIG" outputId="1e13880e-eef5-48cc-8f00-fdee81f555bc"
# Read a Tab Separated File with pandas
tsv_path = "/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/movie_titles_metadata.tsv"
df = pd.read_csv(tsv_path, delimiter="\t")
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="ktwuGRnX44Q5" outputId="a1c25044-7d63-4fbb-d623-f775bfc5cf73"
# Giving the name of the columns
column_names = ["sl.no", "name", "release_year", "rating", "votes", "genres"]
pd.read_csv(tsv_path, delimiter="\t", names=column_names)
```

<!-- #region id="-9uWw_jO5m5W" -->
## **`index_col` Parameter**
The `index_col` parameter specifies which column(s) should be used as the DataFrame's index. The index is used to uniquely label rows in the DataFrame. By default, pandas assigns a numeric index starting from 0. Specifying `index_col` allows you to use one or more columns as the index.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 237} id="3-hqM8pG57c3" outputId="e3b7c364-f0ed-403d-e3b3-1cc88a3703bf"
pd.read_csv(csv_path, index_col="Id").head()
```

<!-- #region id="F3rS-HCg6JW5" -->
## **`header` Parameter**
The `header` parameter specifies which row should be considered as the header (column names) when reading the CSV file. By default, the first row is treated as the header. If you set header=None, pandas will not use any row as the header, and columns will be labeled with numeric indices. You can also provide an integer row number to use as the header, or a list of integers to skip multiple initial rows.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 313} id="4SY2XG1b6YJ5" outputId="78598b33-0739-4e3e-c8ef-96bedefd8c79"
test_csv_path = r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/test.csv"
pd.read_csv(test_csv_path)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 212} id="V4dhKRjW67KQ" outputId="d556ddc4-1bdf-4c61-d143-ed64fa51a8bc"
# Change the header row
pd.read_csv(test_csv_path, header=1)
```

<!-- #region id="qY2YjnJP7Rb5" -->
## **`usecols` Parameter**
The `usecols` parameter in the pandas `read_csv()` function is used to specify which columns from the CSV file should be read into the DataFrame. This parameter allows you to selectively read only a subset of columns, which can be useful when dealing with large datasets where not all columns are needed or when you want to focus on specific data.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 261} id="NurJ7L427wgU" outputId="a6fbed3c-3800-4d47-a1d5-070f5a66b0d1"
train_csv_path = r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/aug_train.csv"
pd.read_csv(train_csv_path).head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="rZ-MK6mt8FeW" outputId="8c8d7197-0486-4658-8576-87fcbfcf24f3"
# Create a list of required columns
required_columns = ["enrollee_id", "gender", "education_level"]
pd.read_csv(train_csv_path, usecols=required_columns).head()
```

<!-- #region id="CoTv1oEC9OLp" -->
## **`skiprows`/`nrows` Parameter**
 - **`skiprows` Parameter:** <br>
The `skiprows` parameter allows you to specify the number of rows at the beginning of the CSV file to skip while reading. You can pass an integer representing the number of rows to skip or a list of row indices (0-based) that should be skipped.

 - **`nrows` Parameter:** <br>
The `nrows` parameter is used to limit the number of rows read from the CSV file. You can pass an integer representing the maximum number of rows to read.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 530} id="pONxGtPx9huQ" outputId="829ee8eb-bd42-488a-9f3c-e706d7a86a89"
pd.read_csv(train_csv_path)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 278} id="2bLw3HC79uty" outputId="2f7985ba-0cf7-4bbf-f0a6-919a7bdb12cf"
# Skip the rows at index 1 and 3
pd.read_csv(train_csv_path, skiprows=[1, 3]).head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 530} id="S-TSlgIu-WP-" outputId="d3989919-8127-41b7-ea54-49e25047aa21"
# Read only the first 100 rows
pd.read_csv(train_csv_path, nrows=100)
```

<!-- #region id="oXLzWZzC-gQv" -->
## **`encoding` Parameter**
The `encoding` parameter is used to specify the character encoding of the CSV file being read. Character encoding defines how characters are represented as bytes in a file. Different encodings are used for different languages and writing systems.

When reading a CSV file, it's important to use the correct encoding to ensure that the data is interpreted correctly. If you encounter issues where characters are not displayed correctly or you see encoding-related errors, specifying the appropriate encoding can help resolve these problems.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 810} id="IvPcPt7M-4Bq" outputId="69a2d6fe-883e-4317-f244-85bfb3575740"
zomato_csv_path = "/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/zomato.csv"
# pd.read_csv(zomato_csv_path)  # Throws and error
pd.read_csv(zomato_csv_path, encoding="latin-1").head()
```

<!-- #region id="nAhGA_1eAs4F" -->
## **`dtype` Parameter**
The `dtype` parameter allows you to explicitly specify the data types for columns when reading a CSV file into a DataFrame. This can be useful when you want to ensure that specific columns are interpreted with the correct data types.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 261} id="9T-XAHp1A2KP" outputId="2ab6d4c9-588c-4e17-d548-8a90ce8be64c"
df = pd.read_csv(train_csv_path)
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="3obvlAojBAe6" outputId="76362c24-868e-4a8f-d23e-76c90f55ef4e"
# Check the columns information of the dataframe
df.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="LrDr3KB0BMET" outputId="16b31182-74ce-454f-ad41-953d665d44f2"
# Convert the datatype of the target column into integer
df = pd.read_csv(train_csv_path, dtype={"target": "int8"})
df.info()
```

<!-- #region id="NXcKGie9ChGL" -->
## **Handling Dates**
The `parse_dates` parameter allows you to specify columns that should be parsed as datetime objects when reading a CSV file. This can be especially useful when you have date or datetime information in your CSV file that you want to directly interpret as datetime objects in your DataFrame.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 417} id="7u0iFSY0Cv1j" outputId="477f62ac-7520-41a0-ab3c-6c7cee786240"
ipl_csv_path = r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/IPL Matches 2008-2020.csv"
df = pd.read_csv(ipl_csv_path)
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="RlClXO4WC6LV" outputId="b4f372b3-1d64-4924-8137-ba733ab4ae35"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="HmjAURzIDCHF" outputId="4a5efc48-283a-40f5-902c-92b71b0c86d9"
df = pd.read_csv(ipl_csv_path, parse_dates=["date"])
df.info()
```

<!-- #region id="yteIcgdYDZix" -->
## **`converters` Parameter**
The `converters` parameter in the pandas `read_csv()` function is used to provide a dictionary of functions that allow you to customize the way specific columns are converted or transformed during the reading process. It's a powerful tool when you need more control over how the data is processed as it's being read from the CSV file.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 417} id="s_GHvt29D7wU" outputId="c6daf762-e764-4aec-a918-151c4116a665"
df = pd.read_csv(ipl_csv_path)
df.head()
```

```python id="5yMS-SAdEI1N"
# Create a function to convert the name of the team into its acronym
def renameTeam(name):
    if name == "Kolkata Knight Riders":
        return "KKR"
    else:
        return name
```

```python colab={"base_uri": "https://localhost:8080/", "height": 417} id="03NToR88ElNI" outputId="11a2f997-1f2f-4fa2-a04a-92862865bb20"
# Apply converters
df = pd.read_csv(ipl_csv_path, converters={"team2": renameTeam})
df.head()
```

<!-- #region id="xmwXvz1kFHE1" -->
## **`na_values` Parameters**
The `na_values` parameter is used to specify a list of values that should be treated as missing or NaN (Not a Number) values when reading a CSV file into a DataFrame. This parameter allows you to define how specific values in your CSV file should be interpreted as missing data.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 261} id="ihKFlrgjFQVV" outputId="a25771e7-5323-4c88-e043-dadc59887048"
pd.read_csv(train_csv_path).head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 261} id="fzK175vuFZLv" outputId="3ce0aa41-2256-4d15-9824-67fd3f140eba"
# Select 'Male' values as 'NaN' for example
pd.read_csv(train_csv_path, na_values="Male").head()
```

<!-- #region id="qDsjpv4GFyyJ" -->
## **Load a Huge Dataset in Chunks**
The `chunksize` parameter in the pandas `read_csv()` function is used to read a large CSV file in smaller, manageable chunks rather than reading the entire file into memory at once. This can be very useful when dealing with datasets that are too large to fit entirely in memory.
<!-- #endregion -->

```python id="_6EOFQqeF8S6"
dfs = pd.read_csv(train_csv_path, chunksize=5000)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Rb2-W65RGHVr" outputId="75983d3b-fead-4912-fce4-cba366de806b"
for chunk in dfs:
    print(chunk.shape)
```

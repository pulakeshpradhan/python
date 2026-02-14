[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/04_Introduction_to_Pandas/04_Read_Write_Excel_CSV.ipynb)

# Read and Write Excel CSV
Pandas is a powerful tool for reading and writing data in various formats including Excel and CSV. In this module, we will explore how to read and write Excel and CSV files using Pandas.


```python
import pandas as pd
```

## 01. Read CSV File Using read_csv() Method
To read a CSV file using Pandas, you can use the read_csv() function. This function takes the filename as an argument and returns a Pandas DataFrame object.


```python
filepath = "D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\stock_data.csv"
df = pd.read_csv(filepath)
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
      <th>tickets</th>
      <th>eps</th>
      <th>revenue</th>
      <th>price</th>
      <th>people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>27.82</td>
      <td>87</td>
      <td>845</td>
      <td>larry page</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WMT</td>
      <td>4.61</td>
      <td>484</td>
      <td>65</td>
      <td>n.a.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>-1</td>
      <td>85</td>
      <td>64</td>
      <td>bill gates</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIL</td>
      <td>not available</td>
      <td>50</td>
      <td>1023</td>
      <td>mukesh ambani</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TATA</td>
      <td>5.6</td>
      <td>-1</td>
      <td>n.a.</td>
      <td>ratan tata</td>
    </tr>
  </tbody>
</table>
</div>



## 02. Skip Rows in DataFrame Using skiprows() Method
In Pandas, you can use the skiprows() method to skip rows in a DataFrame while reading a CSV or Excel file. This can be useful when you have header rows, comment lines, or other non-data rows that you want to exclude from the DataFrame.


```python
df1 = pd.read_csv(filepath, skiprows=1)
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
      <th>GOOGL</th>
      <th>27.82</th>
      <th>87</th>
      <th>845</th>
      <th>larry page</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WMT</td>
      <td>4.61</td>
      <td>484</td>
      <td>65</td>
      <td>n.a.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSFT</td>
      <td>-1</td>
      <td>85</td>
      <td>64</td>
      <td>bill gates</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RIL</td>
      <td>not available</td>
      <td>50</td>
      <td>1023</td>
      <td>mukesh ambani</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TATA</td>
      <td>5.6</td>
      <td>-1</td>
      <td>n.a.</td>
      <td>ratan tata</td>
    </tr>
  </tbody>
</table>
</div>



## 03. Import Data from CSV with "null header"
Sometimes you may encounter CSV files that do not have a header row or have a header row with blank or null values. In Pandas, you can still import such CSV files and specify column names later using the header parameter in the read_csv() function.


```python
df2 = pd.read_csv(filepath, skiprows=1, header=None)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>27.82</td>
      <td>87</td>
      <td>845</td>
      <td>larry page</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WMT</td>
      <td>4.61</td>
      <td>484</td>
      <td>65</td>
      <td>n.a.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>-1</td>
      <td>85</td>
      <td>64</td>
      <td>bill gates</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIL</td>
      <td>not available</td>
      <td>50</td>
      <td>1023</td>
      <td>mukesh ambani</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TATA</td>
      <td>5.6</td>
      <td>-1</td>
      <td>n.a.</td>
      <td>ratan tata</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3 = pd.read_csv(filepath, skiprows=1, header=None, names=["tickets", "eps", "revenue", "price", "people"])
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
      <th>tickets</th>
      <th>eps</th>
      <th>revenue</th>
      <th>price</th>
      <th>people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>27.82</td>
      <td>87</td>
      <td>845</td>
      <td>larry page</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WMT</td>
      <td>4.61</td>
      <td>484</td>
      <td>65</td>
      <td>n.a.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>-1</td>
      <td>85</td>
      <td>64</td>
      <td>bill gates</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIL</td>
      <td>not available</td>
      <td>50</td>
      <td>1023</td>
      <td>mukesh ambani</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TATA</td>
      <td>5.6</td>
      <td>-1</td>
      <td>n.a.</td>
      <td>ratan tata</td>
    </tr>
  </tbody>
</table>
</div>



## 04. Reading Limited Data from CSV
In Pandas, you can read a limited number of rows from a CSV file using the nrows parameter in the read_csv() function.


```python
df4 = pd.read_csv(filepath, nrows=4)
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
      <th>tickets</th>
      <th>eps</th>
      <th>revenue</th>
      <th>price</th>
      <th>people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>27.82</td>
      <td>87</td>
      <td>845</td>
      <td>larry page</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WMT</td>
      <td>4.61</td>
      <td>484</td>
      <td>65</td>
      <td>n.a.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>-1</td>
      <td>85</td>
      <td>64</td>
      <td>bill gates</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIL</td>
      <td>not available</td>
      <td>50</td>
      <td>1023</td>
      <td>mukesh ambani</td>
    </tr>
  </tbody>
</table>
</div>



## 05. Clean Up Messy Data from CSV using na_values() Method
When working with CSV files, you may encounter missing or null values that can make your data messy and difficult to work with. In Pandas, you can use the na_values() method to clean up messy data by specifying which values should be treated as null values.


```python
df5 = pd.read_csv(filepath, na_values=["not available", "n.a."])
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
      <th>tickets</th>
      <th>eps</th>
      <th>revenue</th>
      <th>price</th>
      <th>people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>27.82</td>
      <td>87</td>
      <td>845.0</td>
      <td>larry page</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WMT</td>
      <td>4.61</td>
      <td>484</td>
      <td>65.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>-1.00</td>
      <td>85</td>
      <td>64.0</td>
      <td>bill gates</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIL</td>
      <td>NaN</td>
      <td>50</td>
      <td>1023.0</td>
      <td>mukesh ambani</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TATA</td>
      <td>5.60</td>
      <td>-1</td>
      <td>NaN</td>
      <td>ratan tata</td>
    </tr>
  </tbody>
</table>
</div>



In addition to using a list to specify which values should be treated as null values when reading a CSV file in Pandas, you can also use a dictionary to map specific null values to specific columns. This can be helpful when you need to treat different columns differently based on the null values they contain.


```python
df6 = pd.read_csv(filepath, na_values={
    "revenue": [-1, "n.a.", "not applicable"],
    "eps": ["n.a.", 'not available'],
    "people": ["n.a."],
    "price": ["n.a.", "not available"]
})
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
      <th>tickets</th>
      <th>eps</th>
      <th>revenue</th>
      <th>price</th>
      <th>people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>27.82</td>
      <td>87.0</td>
      <td>845.0</td>
      <td>larry page</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WMT</td>
      <td>4.61</td>
      <td>484.0</td>
      <td>65.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>-1.00</td>
      <td>85.0</td>
      <td>64.0</td>
      <td>bill gates</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIL</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>1023.0</td>
      <td>mukesh ambani</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TATA</td>
      <td>5.60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ratan tata</td>
    </tr>
  </tbody>
</table>
</div>



## 06. Write DataFrame into CSV Using to_csv() Method
Once you have cleaned up and processed your data in a Pandas DataFrame, you may want to save it to a CSV file for further analysis or sharing with others. You can easily do this using the to_csv() method in Pandas.


```python
df6.to_csv("new_stock_data.csv")
```

### 01. index Parameter
In the above code, we use the to_csv() method to write the DataFrame to a CSV file called 'new_stock_data.csv'. We can pass index=False to exclude the DataFrame index from being written to the file.


```python
df6.to_csv("new_stock_data.csv", index=False)
```

### 02. columns Parameter
The to_csv() method in Pandas allows you to customize the output of your DataFrame to a CSV file. One of the options you can specify is the columns parameter, which allows you to write only specific columns from your DataFrame to the CSV file.


```python
df6.columns
```




    Index(['tickets', 'eps', 'revenue', 'price', 'people'], dtype='object')




```python
df6.to_csv("new_stock_data.csv", index=False, columns=["tickets", "eps"])
```

### 03. header Parameter
The header parameter allows you to include or exclude the column names as the first row in the CSV file.


```python
df6.to_csv("new_stock_data.csv", index=False, header=False)
```

## 07. Read Excel File Using read_excel() Method
To read an excel file using Pandas, you can use the read_excel() function. This function takes the filename as an argument and returns a Pandas DataFrame object.


```python
filepath = "D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\stock_data.xlsx"
df7 = pd.read_excel(filepath, sheet_name="stock_data")
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
      <th>tickets</th>
      <th>eps</th>
      <th>revenue</th>
      <th>price</th>
      <th>people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>27.82</td>
      <td>87</td>
      <td>845</td>
      <td>larry page</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WMT</td>
      <td>4.61</td>
      <td>484</td>
      <td>65</td>
      <td>n.a.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>-1</td>
      <td>85</td>
      <td>64</td>
      <td>bill gates</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIL</td>
      <td>not available</td>
      <td>50</td>
      <td>1023</td>
      <td>mukesh ambani</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TATA</td>
      <td>5.6</td>
      <td>-1</td>
      <td>n.a.</td>
      <td>ratan tata</td>
    </tr>
  </tbody>
</table>
</div>



## 08. Converters Argument in read_excel()
The read_excel() function in Pandas allows you to read data from an Excel file into a Pandas DataFrame. One of the arguments you can use to customize the import process is the converters parameter.

The converters parameter is used to specify a dictionary of functions that should be applied to specific columns during the import process. The keys of the dictionary represent the column names or indices, and the values are the functions to apply to the corresponding columns.

In this example, we define a custom function 'convert_people_cell()' that converts any 'n.a.' input to a string which is 'bill gates'. We then read an Excel file called data.xlsx using the read_excel() function and pass a dictionary to the converters parameter. The dictionary has one key-value pair, where the key is the name of the column to apply the function to (people), and the value is the function to apply (convert_people_cell).


```python
def convert_people_cell(cell):
    if cell == "n.a.":
        return "jeff bezos"
    else:
        return cell
```


```python
df8 = pd.read_excel(filepath, converters={
    "people": convert_people_cell
})
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
      <th>tickets</th>
      <th>eps</th>
      <th>revenue</th>
      <th>price</th>
      <th>people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>27.82</td>
      <td>87</td>
      <td>845</td>
      <td>larry page</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WMT</td>
      <td>4.61</td>
      <td>484</td>
      <td>65</td>
      <td>jeff bezos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>-1</td>
      <td>85</td>
      <td>64</td>
      <td>bill gates</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIL</td>
      <td>not available</td>
      <td>50</td>
      <td>1023</td>
      <td>mukesh ambani</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TATA</td>
      <td>5.6</td>
      <td>-1</td>
      <td>n.a.</td>
      <td>ratan tata</td>
    </tr>
  </tbody>
</table>
</div>




```python
def convert_eps_cell(cell):
    if cell == "not available":
        return None
    else:
        return cell
```


```python
df9 = pd.read_excel(filepath, converters={
    "eps": convert_eps_cell,
    "people": convert_people_cell
})
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
      <th>tickets</th>
      <th>eps</th>
      <th>revenue</th>
      <th>price</th>
      <th>people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>27.82</td>
      <td>87</td>
      <td>845</td>
      <td>larry page</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WMT</td>
      <td>4.61</td>
      <td>484</td>
      <td>65</td>
      <td>jeff bezos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>-1.00</td>
      <td>85</td>
      <td>64</td>
      <td>bill gates</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RIL</td>
      <td>NaN</td>
      <td>50</td>
      <td>1023</td>
      <td>mukesh ambani</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TATA</td>
      <td>5.60</td>
      <td>-1</td>
      <td>n.a.</td>
      <td>ratan tata</td>
    </tr>
  </tbody>
</table>
</div>



## 09. Write DataFrame into 'excel' File using to_excel() Method
To write a Pandas DataFrame to an Excel file, you can use the to_excel() method.


```python
df9.to_excel("new_stocks.xlsx", sheet_name="Stocks")
```

### 01. index Parameter
The to_excel() method in Pandas allows you to write a DataFrame to an Excel file with various options to customize the output. One of these options is the index parameter, which controls whether or not to include the DataFrame's index in the Excel file.


```python
df9.to_excel("new_stocks.xlsx", sheet_name="Stocks", index=False)
```

### 02. startrow and startcol Parameter
The startrow and startcol parameters in the to_excel() method of Pandas allow you to specify the starting row and column for writing data to an Excel file.


```python
df9.to_excel("new_stocks.xlsx", sheet_name="Stocks", index=False, startrow=1, startcol=1)
```

## 10. Use ExcelWritter() Class
The ExcelWriter class in Pandas is a powerful tool for writing data frames to one or more sheets in an Excel file. This class provides a lot of flexibility and options for formatting the output, such as specifying the sheet name, adding headers and footers, setting column widths and row heights, and so on.


```python
# Creating two separate dataframe
df_stocs = pd.DataFrame({
    "tickets": ["GOOGLE", "WMT", "MSFT"],
    "price": [845, 65, 64],
    "Pe": [30.37, 14.26, 30.97],
    "eps": [27.82, 4.61, 2.12]
})

df_weather = pd.DataFrame({
    "day": ["1/1/2020", "1/2/2020", "1/3/2020"],
    "temperature": [32, 35, 28],
    "event": ["Rain", "Sunny", "Snow"]
})
```


```python
with pd.ExcelWriter("stocks_and_weather.xlsx") as writer:
    df_stocs.to_excel(writer, sheet_name="Stock", index=False)
    df_weather.to_excel(writer, sheet_name="Weather", index=False)
```

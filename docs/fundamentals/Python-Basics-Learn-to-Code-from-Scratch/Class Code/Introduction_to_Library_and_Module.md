[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Introduction_to_Library_and_Module.ipynb)

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

# Introduction to Library and Modules

```python
# !pip install matplotlib
```

## Import Any Python Library

```python
import numpy as np
import pandas as pd
```

## NumPy


### Creation of One-Dimensional Arrays

```python
myList = [1, 2, 3, 4]
type(myList)
```

```python
myArray = np.array([1, 2, 3, 4], dtype="int8")
type(myArray)
```

```python
myArray.ndim
```

```python
myArray.dtype
```

```python
# Accesing elements of an one-dimensional array
myArray[2]
```

 ### Creation of Two-Dimensional Arrays

```python
myList = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

```python
myArray2D = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="int8")
```

```python
myArray2D
```

```python
myArray2D.dtype
```

```python
myArray2D.ndim
```

```python
myArray2D.shape
```

```python
myArray2D[1, 1]
```

```python
myArray2D[2, 2]
```

## Pandas

```python
# Read CSV File using pandas
file_path = r"D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\DailyDelhiClimateTest.csv"
```

```python
df = pd.read_csv(file_path)
type(df)
```

```python
df.head()
```

```python
df.tail()
```

```python
df.head(10)
```

```python
df.shape
```

```python
df.ndim
```

```python
df.dtypes
```

```python
new_df = df[["date", "meantemp", "humidity"]][df["meantemp"] > 30]
```

```python
new_df.shape
```

```python
new_df.head()
```

```python
meantemp = new_df["meantemp"]
```

```python
type(meantemp)
```

```python
humidity = new_df["humidity"]
```

```python
type(humidity)
```

## Plotting the Data

```python
import matplotlib.pyplot as plt
```

```python
plt.scatter(x=meantemp, y=humidity, c="red")
plt.xlabel("Mean Temperature (in C)")
plt.ylabel("Humidity")
plt.title("Scatterplot between Mean Temp and Humidity")
plt.grid()
```

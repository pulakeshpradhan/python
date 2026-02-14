[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/00_Data_Gathering/04_Understanding_Your_Data.ipynb)

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

# **Understanding Your Data**
Understanding the data is a critical step in the data science process. It involves gaining insights into the structure, content, quality, and characteristics of the data you're working with. Properly understanding the data sets the foundation for making informed decisions, building accurate models, and deriving meaningful insights.


## **Import Required Libraries**

```python
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
```

## **Read the Data**

```python
df = pd.read_csv(r"D:\Coding\Datasets\Global YouTube Statistics.csv", encoding="latin")
```

## **How Big is the Data?**

```python
df.shape
```

## **How does the Data look like?**

```python
# Print first 5 rows of the dataframe
df.head()
```

```python
# Randomly choose 5 rows and print it
df.sample(5)
```

## **What are the Data Types of the Columns?**

```python
df.info()
```

## **Are there any Missing Values?**

```python
# Checking the number of missing values for each column
df.isnull().sum()
```

## **How does the Data look Mathematically?**

```python
df.describe()
```

## **Are there any Duplicate Rows?**

```python
df.duplicated().sum()
```

## **How is the Correlation between Columns?**

```python
# Extract the correlation between all the variables
df.corr()
```

```python
# Extract the correlation between the 'subscribers' and other numerical columns
df.corr()["subscribers"]
```

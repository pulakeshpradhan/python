[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/03_One_Hot_Encoding.ipynb)

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

# **One Hot Encoding**
One-Hot Encoding is a popular technique used in machine learning and data preprocessing, especially when dealing with categorical data. It is used to represent categorical variables as binary vectors or matrices, where each category is mapped to a unique binary value. 

This transformation is necessary because many machine learning algorithms and models require numerical input, and categorical data in its raw form cannot be directly used in these algorithms.


<img src="https://miro.medium.com/v2/resize:fit:1358/1*ggtP4a5YaRx6l09KQaYOnw.png">


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## **Read the Data**

```python
df = pd.read_csv("D:\Coding\Datasets\cars.csv")
df.head()
```

```python
df.shape
```

```python
# Check the number of unique brand names
df["brand"].nunique()
```

```python
# Count the values for each brand in 'brand' column
df["brand"].value_counts()
```

```python
# Count the values for each unique name in 'fuel' column
df["fuel"].value_counts()
```

```python
# Count the values for each unique name in 'owner' column
df["owner"].value_counts()
```

## **One Hot Encoding with Pandas**

```python
# Applying One Hot Encoding on 'fuel' and 'owner' columns
pd.get_dummies(data=df, columns=["fuel", "owner"])
```

## **K-1 One Hot Encoding with Pandas**


When using the pd.get_dummies() function in Pandas, you can drop the first category (column) of each categorical variable to avoid multicollinearity, which can be useful in certain situations. This is done using the drop_first parameter. Setting drop_first=True will drop the first category from each categorical variable after one-hot encoding.

```python
# Applying One Hot Encoding on 'fuel' and 'owner' columns
# Removing the first categorical variable to avoid multicolinearity
pd.get_dummies(data=df, columns=["fuel", "owner"], drop_first=True)
```

## **One Hot Encoding using Sklearn**


### **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
# Print the dataframe
df.head()
```

```python
x_train, x_test, y_train, y_test = train_test_split(df.drop("selling_price", axis=1),
                                                    df["selling_price"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, y_train.shape
```

```python
x_train.head()
```

```python
x_train.shape
```

### **Apply OHE on 'fuel' and 'owner' Columns**

```python
from sklearn.preprocessing import OneHotEncoder
```

```python
# Creating an object of the One Hot Encode class
one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False, dtype=np.int8)

# Separating the 'fuel' and 'owner' columns from the x_train dataframe
# Fit the separated training data
one_hot_encoder.fit(x_train[["fuel", "owner"]])

# Transform the separated training data
x_train_encoded = one_hot_encoder.transform(x_train[["fuel", "owner"]])
x_train_encoded
```

```python
x_train_encoded.shape
```

```python
# Merge the x_train_encoded columns with the 'brand' and 'km_driven' columns
x_train_merged = np.hstack((x_train[["brand", "km_driven"]], x_train_encoded))
x_train_merged
```

```python
x_train_merged.shape
```

```python
# Print the column names of the encoded x_train data
one_hot_encoder.get_feature_names_out()
```

```python
# Define the column names in an array
column_names = np.concatenate((x_train.columns[0:2], one_hot_encoder.get_feature_names_out()), axis=0)
print(len(column_names))
column_names
```

```python
# Convert the x_train_merged array into pandas dataframe
x_train_encoded = pd.DataFrame(data=x_train_merged, columns=column_names)
x_train_encoded
```

```python
x_train_encoded.shape
```

```python
# Print the x_test data
x_test.head()
```

```python
# Encode x_test data
x_test_encoded = one_hot_encoder.transform(x_test[["fuel", "owner"]])
x_test_encoded
```

```python
# Merge the x_test_encoded columns with the 'brand' and 'km_driven' columns
x_test_merged = np.hstack((x_test.iloc[:, 0:2], x_test_encoded))
x_test_merged
```

```python
# Convert the x_test_merged array into pandas dataframe
x_test_encoded = pd.DataFrame(data=x_test_merged, columns=column_names)
x_test_encoded
```

## **Apply OHE on 'brand' Column using Pandas**

```python
# Count the values for each brand in 'brand' column
counts = df["brand"].value_counts()
counts
```

```python
# Check the total number of unique brands
df["brand"].nunique()
```

```python
# Define a threshold
threshold = 100
```

```python
# Store the name of brands in a list where the value count is less than 100
repl = counts[counts <= threshold].index
repl
```

```python
# Replace the name of the brand with 'others'
new_df = df.replace(to_replace=repl, value="Others")
new_df
```

```python
new_df["brand"].value_counts()
```

```python
# Apply OHE on 'brand' column of the new dataframe
pd.get_dummies(data=new_df["brand"]).sample(20)
```

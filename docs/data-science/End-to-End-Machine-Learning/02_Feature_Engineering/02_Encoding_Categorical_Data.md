[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/02_Encoding_Categorical_Data.ipynb)

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

# **Encoding Categorical Data**
Encoding categorical data is an essential step in preparing data for machine learning models since most machine learning algorithms require numerical input data. Categorical data represents non-numeric data such as categories, labels, or classes.

In Python, you can use various techniques to encode categorical data, and the choice of encoding method depends on the nature of your data and the machine learning algorithm you plan to use.


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## **Read the Data**

```python
df = pd.read_csv("D:\Coding\Datasets\customer.csv")
df.head()
```

```python
df.shape
```

```python
# Extrcting the 'review', 'education' and 'purchased' colums from the dataframe
df = df.iloc[:, 2:]
```

```python
df.head()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
x_train, x_test, y_train, y_test = train_test_split(df.drop("purchased", axis=1),
                                                    df["purchased"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

## **Ordinal Encoding**
Ordinal encoding is a technique for encoding categorical data where the categories have a meaningful order or ranking. This method assigns a unique integer value to each category based on its order or priority. Ordinal encoding is appropriate when the categorical data represents ordered or ranked values, such as "low," "medium," and "high" or "small," "medium," "large."

```python
from sklearn.preprocessing import OrdinalEncoder
```

```python
# Checking the unique values in each column
print("Unique values in each column:")
for i in range(len(df.columns)):
    print(f"{df.columns[i]}: {df.iloc[:, i].unique()}")
```

```python
# Creating an object of ordinal encoder class
ordinal_encoder = OrdinalEncoder(categories=[["Poor", "Average", "Good"], ["School", "UG", "PG"]],
                                 dtype=np.int8)
# Fit the training data
ordinal_encoder.fit(x_train)

# Transform the training and testing data
x_train_encoded = ordinal_encoder.transform(x_train)
x_test_encoded = ordinal_encoder.transform(x_test)
```

```python
ordinal_encoder.categories_
```

```python
# Converting the encoded array into pandas dataframe
x_train_encoded = pd.DataFrame(x_train_encoded, columns=["review", "education"])
x_test_encoded = pd.DataFrame(x_test_encoded, columns=["review", "eucation"])
```

```python
# Print the non-encoded training data
x_train.head()
```

```python
# Print the encoded training data
x_train_encoded.head()
```

```python
# Print the encoded testing data
x_test_encoded.head()
```

## **Label Encoding**
Label encoding is a technique for encoding categorical data into numerical values, where each category is assigned a unique integer label. This encoding is suitable for categorical data where there is no inherent order or ranking among the categories.

You can use the **'LabelEncoder'** class from the sklearn.preprocessing module to perform label encoding. This encode target labels with value between 0 and n_classes-1. This transformer should be used to encode target values, i.e. y and not the input X.

```python
from sklearn.preprocessing import LabelEncoder
```

```python
# Creating an object of the label encoder class
label_encoder = LabelEncoder()

# Fit the training data
label_encoder.fit(y_train)

# Transform the training and testing data
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
```

```python
label_encoder.classes_
```

```python
# Print the y_train data
y_train.head(10)
```

```python
# Print the y_train_encoded data
y_train_encoded
```

```python
# Print the y_test_encoded data
y_test_encoded
```

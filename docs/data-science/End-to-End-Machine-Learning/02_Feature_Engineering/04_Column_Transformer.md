[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/04_Column_Transformer.ipynb)

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

# **Column Transformer**
The ColumnTransformer is a feature in scikit-learn, a popular Python machine learning library, that allows you to apply different preprocessing steps to different subsets of the columns (features) in your dataset. It is particularly useful when you have a dataset with a mix of numerical and categorical features, and you want to apply different transformations to these feature types.

Here's an overview of how the ColumnTransformer works:

1. **Specify Transformers:**<br> First, you define a list of transformers, where each transformer specifies a particular preprocessing step to be applied to a subset of the columns. For example, you might have one transformer for numerical columns (e.g., scaling), another for categorical columns (e.g., one-hot encoding), and maybe even other transformers for specific subsets of columns.

2. **Specify Columns:**<br> For each transformer, you also specify which columns it should be applied to. This is done using the columns parameter, where you can specify either column indices or column names.

3. **Combine Transformers:**<br> You create a ColumnTransformer object and pass in the list of transformers. You can also specify what to do with the remaining columns that are not specified in any of the transformers, using the remainder parameter. Options include dropping them or passing them through without any transformation.

4. **Fit and Transform:**<br> You can then fit the ColumnTransformer on your dataset using the fit method, and subsequently transform your dataset using the transform method. The ColumnTransformer applies the specified transformations to the designated columns and returns a transformed dataset.


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
```

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
```

## **Read the Data**

```python
df = pd.read_csv("D:\Coding\Datasets\covid_toy.csv")
df
```

```python
# Check the information of the columns
df.info()
```

```python
# Check the number of null values in each column
df.isnull().sum()
```

```python
# Check all the unique values of the categorical columns
for column in df.select_dtypes(include="object").columns:
    unique_values = df[column].unique()
    print(f"{column}: {unique_values}")
```

## **Preprocessing without Column Transformer**


### **Train Test Split**

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop("has_covid", axis=1),
                                                    df["has_covid"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

```python
x_train.head(10)
```

### **Fill the Null Values of 'fever' Column using SimpleImputer**

```python
# Create a SimpleImputer object
simple_imputer = SimpleImputer()

# Fit the 'fever' column of the training data
simple_imputer.fit(x_train[["fever"]])

# Transform the 'fever' column of the training and testing data
x_train_fever = simple_imputer.transform(x_train[["fever"]])
x_test_fever = simple_imputer.transform(x_test[["fever"]])
```

```python
# Print the first ten values of the x_train_fever
x_train_fever[:10]
```

### **Apply Ordinal Encdoing to 'cough' Column**

```python
# Create an object of the OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[["Mild", "Strong"]], dtype=int)

# Fit the 'cough' column of the training data
ordinal_encoder.fit(x_train[["cough"]])

# Transform the 'cough' column of the training and testing data
x_train_cough = ordinal_encoder.transform(x_train[["cough"]])
x_test_cough = ordinal_encoder.transform(x_test[["cough"]])
```

```python
# Print the first ten values of the x_train_cough
x_train_cough[:10]
```

### **Apply One Hot Encdoing to 'gender' and 'city' Columns**

```python
# Create an object of the OneHotencoder
one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False, dtype=int)

# Fit the 'genedr' and 'city' columns of the training data
one_hot_encoder.fit(x_train[["gender", "city"]])

# Transform the 'genedr' and 'city' columns of the training and testing data
x_train_gender_city = one_hot_encoder.transform(x_train[["gender", "city"]])
x_test_gender_city = one_hot_encoder.transform(x_test[["gender", "city"]])
```

```python
# Check the new column names after applying One Hot Encoding
one_hot_encoder.get_feature_names_out()
```

```python
# Print the first ten values of the x_train_gender_city
x_train_gender_city[:10]
```

```python
x_train_cough.shape
```

```python
# Convert the 'age' column into numpy array 
x_train_age = np.array(x_train["age"]).reshape((70, 1))
x_test_age = np.array(x_test["age"]).reshape((30, 1))
```

```python
# Print the first ten values of the x_train_age
x_train_age[:10]
```

### **Concatenating all the Arrays for the Training and Testing Data**

```python
# Concatenating all the columns of the training data
x_train_transformed = np.concatenate((x_train_age, x_train_fever, x_train_cough, x_train_gender_city), axis=1)

# Concatenating all the columns of the training data
x_test_transformed = np.concatenate((x_test_age, x_test_fever, x_test_cough, x_test_gender_city), axis=1)
```

```python
# Defining the column names of the transformed dataframe
column_names = np.concatenate((np.array(["age", "fever", "cough"]), one_hot_encoder.get_feature_names_out()))
column_names
```

```python
# Convert transformed data into pandas dataframe
x_train_transformed = pd.DataFrame(x_train_transformed, columns=column_names)
x_test_transformed = pd.DataFrame(x_test_transformed, columns=column_names)
```

```python
# Print the transformed data
x_train_transformed
```

```python
# Print the information of the transformed training data
x_train_transformed.info()
```

```python
x_test_transformed.head(10)
```

```python
# Print the information of the transformed testing data
x_test_transformed.info()
```

## **Preprocessing with Column Transformer**

```python
from sklearn.compose import ColumnTransformer
```

```python
# Create an object of the ColumnTransformer
transformer = ColumnTransformer(transformers=[
    ("tranformer_1", SimpleImputer(), ["fever"]),
    ("transformer_2", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
    ("transformer_3", OneHotEncoder(drop="first", sparse_output=False), ["gender", "city"])
], remainder="passthrough")
```

```python
# Fit and transform the training data
x_train_transformed = transformer.fit_transform(x_train)
x_train_transformed.shape
```

```python
# Transform the testing data
x_test_transformed = transformer.transform(x_test)
x_test_transformed.shape
```

```python
# Checking the new column names of the transformed data
transformer.get_feature_names_out()
```

```python
# Convert the transformed array into pandas dataframe
x_train_transformed = pd.DataFrame(x_train_transformed, columns=transformer.get_feature_names_out())
x_test_transformed = pd.DataFrame(x_test_transformed, columns=transformer.get_feature_names_out())
```

```python
x_train_transformed
```

```python
x_test_transformed.head(10)
```

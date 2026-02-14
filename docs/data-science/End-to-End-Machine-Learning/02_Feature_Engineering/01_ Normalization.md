[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/01_ Normalization.ipynb)

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

# **Normalization - Feature Scaling**
Normalization, also known as Min-Max scaling, is a feature scaling technique used in data preprocessing to rescale numerical features to a specific range, typically between 0 and 1. The goal of normalization is to ensure that all feature values have similar scales, making them more suitable for machine learning algorithms that are sensitive to the magnitude of input features.


<center><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTBOZqDiNcILVqNqUPmDj8r3I7f_GYD9Op6qrjK1BB8r5iKASSzZoyWOTv4HE4V2JVYGx0&usqp=CAU"></ceneter>


## **When to use Normalization?**


<center><img src="https://miro.medium.com/max/1400/1*qRmiffZgkNaXnTBZwDafCA.png" style="max-width:800px"></center>


# **Example**


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## **Read the Data**

```python
# Read some specific column from the data
csv_path = r"D:\Coding\Datasets\wine_data.csv"
df = pd.read_csv(csv_path, usecols=["Class", "Alcohol", "Malic acid"])
df.head()
```

```python
df.shape
```

## **Data Visualization**

```python
# Creating probabilty density plots of the data
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

sns.kdeplot(data=df["Alcohol"], ax=ax1, c="red", label="Alcohol")
ax1.set_title("Probability Distribution of Alcohol")
ax1.legend()

sns.kdeplot(data=df["Malic acid"], ax=ax2, c="blue", label="Malic acid")
ax2.set_title("Probability Distribution of Malic acid")
ax2.legend()

plt.show()
```

```python
# Creating a scatter plot of the data
sns.scatterplot(x=df["Alcohol"], y=df["Malic acid"], hue=df["Class"], palette=["red", "blue", "green"])
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop("Class", axis=1), 
                                                    df["Class"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

## **MinMax Scaler**

```python
from sklearn.preprocessing import MinMaxScaler

# Creating a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the train set, it will learn the parameters
scaler.fit(x_train)

# Transform train and test sets
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

```python
# transform always returns a numpy array
# Converting the scaled numpy arrays into pandas dataframes
x_train_scaled = pd.DataFrame(data=x_train_scaled, columns=["Alcohol", "Malic acid"])
x_test_scaled = pd.DataFrame(data=x_test_scaled, columns=["Alcohol", "Malic acid"])
```

```python
x_train_scaled.head()
```

```python
# Describe the training data
np.round(x_train.describe(), 2)
```

```python
# Describe the scaled training data
np.round(x_train_scaled.describe(), 2)
```

## **Effect of Scaling**

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a scatter plot of the training data
sns.scatterplot(data=x_train, x="Alcohol", y="Malic acid", ax=ax1, c=y_train)
ax1.set_title("Before Scaling")

# Creating a scatter plot of the scaled training data
sns.scatterplot(data=x_train_scaled, x="Alcohol", y="Malic acid", ax=ax2, c=y_train)
ax2.set_title("After Scaling")

plt.show()
```

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a probability density plot of the training data
sns.kdeplot(data=x_train["Alcohol"], ax=ax1, label="Alcohol")
sns.kdeplot(data=x_train["Malic acid"], ax=ax1, label="Malic acid")
ax1.legend()
ax1.set_xlabel("")
ax1.set_title("Before Scaling")

# Creating a probability density plot of the scaled training data
sns.kdeplot(data=x_train_scaled["Alcohol"], ax=ax2, label="Alcohol")
sns.kdeplot(data=x_train_scaled["Malic acid"], ax=ax2, label="Malic acid")
ax2.legend()
ax2.set_xlabel("")
ax2.set_title("After Scaling")

plt.show()
```

## **Comparison of Distribution**

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a probabilty density plot of the 'Alcohol' column from the training data
sns.kdeplot(x_train["Alcohol"], ax=ax1)
ax1.set_title("Alcohol Distribution before Scaling")

# Creating a probabilty density plot of the 'Alcohol' column from the scaled training data
sns.kdeplot(x_train_scaled['Alcohol'], ax=ax2)
ax2.set_title("Alcohol Distribution after Scaling")

plt.show()
```

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a probabilty density plot of the 'Malic acid' column from the training data
sns.kdeplot(x_train["Malic acid"], ax=ax1)
ax1.set_title("Malic acid Distribution before Scaling")

# Creating a probabilty density plot of the 'Malic acid' column from the scaled training data
sns.kdeplot(x_train_scaled['Malic acid'], ax=ax2)
ax2.set_title("Malic acid Distribution after Scaling")

plt.show()
```

## **Difference between Standardization and Normalization**
Standardization and normalization are two common techniques for feature scaling in data preprocessing, and they have different approaches and effects on the data:

**Goal:**<br>
* **Standardization:** The goal of standardization is to transform the data in such a way that it has a mean of 0 and a standard deviation of 1. It centers the data around zero and scales it by the standard deviation.
* **Normalization:** The goal of normalization is to rescale the data to a specific range, typically between 0 and 1. It preserves the relative relationships between data points but scales them to fit within the chosen range.


<center><img src="https://miro.medium.com/v2/resize:fit:744/1*HW7-kYjj6RKwrO-5WTLkDA.png"></center>

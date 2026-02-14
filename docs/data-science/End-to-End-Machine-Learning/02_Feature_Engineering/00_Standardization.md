[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/00_Standardization.ipynb)

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

# **Standardization - Feature Scaling**
Standardization, also known as z-score normalization or standard scaling, is a technique used in data preprocessing to rescale the features of a dataset. The goal of standardization is to transform the data so that it has a mean of 0 and a standard deviation of 1. This process helps to make the features more comparable and can be particularly useful in machine learning algorithms that are sensitive to the scale of the input features, such as gradient descent-based optimization algorithms.


<center><img src="https://miro.medium.com/v2/resize:fit:552/1*DK6tNx7Ke_27-CdLT3_1Ug.png"></center>


**1. Standard Deviation:** <br>
Standard deviation is a statistical measure that quantifies the amount of variation or dispersion in a set of data points. It provides a way to understand how spread out the values in a dataset are and how far individual data points are from the mean (average). In essence, it tells you whether the data points are clustered closely around the mean or scattered widely.
```
σ (sigma) = √[Σ(xi - μ)² / N]
```


**2. Z-score:**<br>
A Z-score, also known as a standard score, is a statistical measure that quantifies how far away a particular data point is from the mean (average) of a dataset when measured in terms of standard deviations. It's a way to standardize and normalize data, making it easier to compare values from different datasets or different parts of the same dataset.
```
Z = (X - μ) / σ
```


## **When to use Standardization?**


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
df = pd.read_csv("D:\Coding\Datasets\Social_Network_Ads.csv")
df.head()
```

```python
df.shape
```

```python
# Checking the null values
df.isnull().sum()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop("Purchased", axis=1),
                                                    df["Purchased"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

## **Standard Scaler**
In scikit-learn, the StandardScaler is a preprocessing technique provided by the library for standardizing or scaling features in your dataset. It follows the standardization process I described earlier, where it scales the features to have a mean of 0 and a standard deviation of 1. This is done to ensure that all features have the same scale, making them more suitable for machine learning algorithms that are sensitive to feature scales.

The 'fit' method is typically called on a machine learning model or a data preprocessing object to adapt it to the specific dataset you are working with. Its purpose is to learn from the data and update the internal state of the object.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit the scaler to the train set, it will learn the parameters
scaler.fit(x_train)

# Transform train and test sets
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

```python
# transform always returns a numpy array
# Converting the scaled numpy arrays into pandas dataframes
x_train_scaled = pd.DataFrame(data=x_train_scaled, columns=x_train.columns)
x_test_scaled = pd.DataFrame(data=x_test_scaled, columns=x_test.columns)
```

```python
x_train_scaled.head()
```

```python
# Describe the training data
np.round(x_train.describe(), 1)
```

```python
# Describe the scaled training data
np.round(x_train_scaled.describe(), 1)
```

## **Effect of Scaling**

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a scatter plot of the training data
ax1.scatter(x=x_train["Age"], y=x_train["EstimatedSalary"])
ax1.set_title("Before Standardization")
ax1.set_xlabel("Age")
ax1.set_ylabel("Estimated Salary")

# Creating a scatter plot of the scaled training data
ax2.scatter(x=x_train_scaled["Age"], y=x_train_scaled["EstimatedSalary"], color="red")
ax2.set_title("After Standardization")
ax2.set_xlabel("Standardized Age")
ax2.set_ylabel("Standardized Estimated Salary")
plt.show()
```

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a probabilty density plot of the training data
ax1.set_title("Before Standardization")
sns.kdeplot(data=x_train["Age"], ax=ax1, label="Age")
sns.kdeplot(data=x_train["EstimatedSalary"], ax=ax1, label="Estimated Salary")
ax1.legend()

# Creating a probabilty density plot of the scaled training data
ax2.set_title("After Standardization")
sns.kdeplot(data=x_train_scaled["Age"], ax=ax2, label="Age")
sns.kdeplot(data=x_train_scaled["EstimatedSalary"], ax=ax2, label="Estimated Salary")
ax2.legend()

ax1.set_xlabel("")
ax2.set_xlabel("")
plt.show()
```

## **Comparison of Distribution**

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a probabilty density plot of the 'Age' column from the training data
sns.kdeplot(x_train["Age"], ax=ax1)
ax1.set_title("Age Distribution before Scaling")

# Creating a probabilty density plot of the 'Age' column from the scaled training data
sns.kdeplot(x_train_scaled['Age'], ax=ax2)
ax2.set_title("Age Distribution after Scaling")

plt.show()
```

## **Why Scaling is Important?**


### **Comparison on Logistic Regression Model**

```python
from sklearn.linear_model import LogisticRegression
```

```python
# Creating two logistic regression model for the training data and scaled training data respectively
lr = LogisticRegression()
lr_scaled = LogisticRegression()
```

```python
# Fitting the data to the models
lr.fit(X=x_train, y=y_train)
lr_scaled.fit(X=x_train_scaled, y=y_train)
```

```python
# Predict the testing data
y_pred = lr.predict(x_test)
y_pred_scaled = lr_scaled.predict(x_test_scaled)
```

```python
# Calculating the accuracy
from sklearn.metrics import accuracy_score

print("Accuracy Score on Actual Data:", accuracy_score(y_test, y_pred).round(2))
print("Accuracy Score on Scaled Data:", accuracy_score(y_test, y_pred_scaled).round(2))
```

### **Comparison on Decision Tree Model**

```python
from sklearn.tree import DecisionTreeClassifier
```

```python
# Creating two decision tree model for the training data and scaled training data respectively
dt = DecisionTreeClassifier()
dt_scaled = DecisionTreeClassifier()
```

```python
# Fitting the data to the models
dt.fit(X=x_train, y=y_train)
dt_scaled.fit(X=x_train_scaled, y=y_train)
```

```python
# Predict the testing data
y_pred = dt.predict(x_test)
y_pred_scaled = dt_scaled.predict(x_test_scaled)
```

```python
# Calculating the accuracy
print("Accuracy Score on Actual Data:", accuracy_score(y_test, y_pred).round(2))
print("Accuracy Score on Scaled Data:", accuracy_score(y_test, y_pred_scaled).round(2))
```

## **Effect of Outlier**
Standardization, by itself, does not handle outliers in the data. In fact, standardization can sometimes exacerbate the impact of outliers, making them more prominent in the scaled data.

```python
# Adding some outliers to the data
outliers = pd.DataFrame({"Age":[10, 90, 97], "EstimatedSalary":[1000, 250000, 350000], "Purchased": [0, 1, 1]})
outliers
```

```python
new_df = pd.concat([df, outliers])
new_df
```

```python
# Create a scatter plot
plt.scatter(x=new_df["Age"], y=new_df["EstimatedSalary"])
plt.show()
```

### **Applying Standardization**

```python
# Splitting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(new_df.drop(["Purchased"], axis=1), 
                                                    new_df["Purchased"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

```python
# Creating a Standard Scaler object
scaler = StandardScaler()

# Fitting the data
scaler.fit(x_train)

# Applying Standardization on the training and testing data
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

```python
# Converting the scaled data into a pandas dataframe
x_train_scaled = pd.DataFrame(data=x_train_scaled, columns=x_train.columns)
x_test_scaled = pd.DataFrame(data=x_test_scaled, columns=x_test.columns)
x_train_scaled.head()
```

```python
# Plot the data
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Plotting the x_train data
ax1.scatter(x=x_train["Age"], y=x_train["EstimatedSalary"])
ax1.set_title("Before Standardization")
ax1.set_xlabel("Age")
ax1.set_ylabel("Estimated Salary")

# Plotting the scaled x_train data
ax2.scatter(x=x_train_scaled["Age"], y=x_train_scaled["EstimatedSalary"], color="red")
ax2.set_title("After Standardization")
ax2.set_xlabel("Standardized Age")
ax2.set_ylabel("Standardized Estimated Salary")
```

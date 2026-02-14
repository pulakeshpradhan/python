[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/00_Linear Regression/00_Simple_Linear_Regression.ipynb)

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

# **Simple Linear Regression**

Simple linear regression is a statistical method used to model the relationship between two variables, typically denoted as X and Y. It is a straightforward approach to understanding how changes in one variable (X) are associated with changes in another variable (Y). The goal of simple linear regression is to find a linear equation that best represents this relationship.


<center><img src="https://miro.medium.com/v2/resize:fit:1400/1*GSAcN9G7stUJQbuOhu0HEg.png" style="width:50%"></center>


<center><img src="https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704-EP713_MultivariableMethods/SimpleLinearRegression.png" style="width:70%"></center>


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

## **Read the Data**

```python
df = pd.read_csv("D:\Coding\Datasets\Placement_SLR.csv")
df
```

## **Plot the Data**

```python
sns.scatterplot(x=df["cgpa"], y=df["package"])
plt.title("Scatterplot between CGPA and Package")
plt.show()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
x_train, x_test, y_train, y_test = train_test_split(df["cgpa"],
                                                    df["package"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

## **Train a Simple Linear Regression Model**

```python
from sklearn.linear_model import LinearRegression
```

```python
# Instantiate a LinearRegression object
lr = LinearRegression()

# Fit the training data
lr.fit(x_train.values.reshape(140, 1), y_train)
```

```python
# Predict the Test data
y_pred = lr.predict(x_test.values.reshape(60, 1))
y_pred
```

## **Plot the Best Fit Line**

```python
sns.scatterplot(x=df["cgpa"], y=df["package"])
sns.lineplot(x=x_train, y=lr.predict(x_train.values.reshape(140, 1)), c="red",
             label="Regression Line")
plt.title("Scatterplot between CGPA and Package")
plt.show()
```

## **Fetch the Slope and Y-intercept Value**

```python
# Extract the slope value
m = lr.coef_[0]
# Extract the y-intercept value
c = lr.intercept_

print("Slope (m):", m)
print("Y-intercept (c):", c)
```

## **Check the RMSE**

```python
from sklearn.metrics import mean_squared_error
```

```python
# Calculate the Root Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error (MSE):", mse.round(2))
print("Root Mean Squared Error (RMSE):", rmse.round(2))
```

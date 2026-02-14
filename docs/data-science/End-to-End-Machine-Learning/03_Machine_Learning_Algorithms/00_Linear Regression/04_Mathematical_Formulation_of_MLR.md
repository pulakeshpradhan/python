[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/00_Linear Regression/04_Mathematical_Formulation_of_MLR.ipynb)

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

# **Mathematical Formulation of Multiple Linear Regression**
In multiple linear regression, the slope (m) and the intercept (b) of the linear equation can be calculated using the following mathematical formula:


<center><img src="https://i0.wp.com/cmdlinetips.com/wp-content/uploads/2020/03/Linear_Regression_Beta_Hat_Matrix_Multiplication.png?resize=561%2C136&ssl=1" style="width:30%"></center>


Read this Blog:<br>
https://towardsdatascience.com/building-linear-regression-least-squares-with-linear-algebra-2adf071dd5dd


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
# Read the diabetes data from scikit learn
from sklearn.datasets import load_diabetes
```

```python
X, y = load_diabetes(return_X_y=True)
```

```python
X.shape
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape
```

## **Train a Linear Regression Model** 

```python
from sklearn.linear_model import LinearRegression
```

```python
# Instantiate a LinearRegression object
lr = LinearRegression()

# Fit the training data
lr.fit(X_train, y_train)
```

```python
# Print the coefficients
lr.coef_
```

```python
# Print the intercept
lr.intercept_
```

## **Accuracy Assessment**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

```python
# Predict the test data
y_pred = lr.predict(X_test)
```

```python
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

## **Build a Custom Linear Regression Model**

```python
class CustomLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X_train, y_train):
        X = np.insert(X_train, 0, 1, axis=1)
        y = y_train
        betas = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
        
    def predict(self, X_test):
        y_pred = X_test.dot(self.coef_) + self.intercept_
        return y_pred
```

```python
# Instantiate a CustomLR object
lr = CustomLR()

# Fit the training data
lr.fit(X_train, y_train)
```

```python
# Print the coefficients
lr.coef_
```

```python
# Print the intercept
lr.intercept_
```

```python
# Predict the test data
y_pred = lr.predict(X_test)
```

```python
# Check the accuarcy
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

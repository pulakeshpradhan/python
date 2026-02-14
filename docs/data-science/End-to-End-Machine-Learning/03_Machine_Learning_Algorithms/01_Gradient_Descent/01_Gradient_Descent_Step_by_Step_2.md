[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/01_Gradient_Descent/01_Gradient_Descent_Step_by_Step_2.ipynb)

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

# **Gradient Descent: Step by Step 2**


## **Import Required Libraries** 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

## **Make a Data**

```python
from sklearn.datasets import make_regression
```

```python
# Make a data for regression
X, y = make_regression(n_samples=100,
                       n_features=1,
                       n_informative=1,
                       n_targets=1,
                       noise=20,
                       random_state=0)
```

```python
# Plot the data
sns.scatterplot(x=X.flatten(), y=y)
plt.show()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape
```

## **Apply Linear Regression with Ordinary Least Squares (OLS)**

```python
from sklearn.linear_model import LinearRegression
```

```python
# Instantiate an object of the LinearRegression class
lin_reg = LinearRegression()

# Fit the data
lin_reg.fit(X, y)
```

```python
# Print the coefficient value
print("Coefficient (m):", lin_reg.coef_)
```

```python
# Print the intercept value
print("Intercept (b):", lin_reg.intercept_)
```

```python
# Calculate the accuracy on the test data
from sklearn.metrics import r2_score
```

```python
y_pred = lin_reg.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
```

```python
# Plot the regression line
sns.scatterplot(x=X_train.flatten(), y=y_train)
sns.lineplot(x=X_train.flatten(), y=lin_reg.predict(X_train), c="#b10026", label="Regression Line")
plt.show()
```

## **Apply Linear Regression with Gradient Descent**

```python
# Create a class to apply gradient descent
class GDRegressor:
    def __init__(self, learning_rate, epochs):
        self.m = 100
        self.b = -120
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def fit(self, X, y):
        # Calculating the slope(m) and intercept(b) using GD
        for i in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m*X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((y - self.m*X.ravel() - self.b)*X.ravel())
            
            self.b = self.b - (self.learning_rate * loss_slope_b)
            self.m = self.m - (self.learning_rate * loss_slope_m)
            
        print("Coefficient (m):", self.m)
        print("Intercept (b):", self.b)
        
    def predict(self, X):
        return self.m * X + self.b
```

```python
# Instantiate a GDRegressor object
gd = GDRegressor(learning_rate=0.001, epochs=50)

# Fit the data
gd.fit(X_train, y_train)
```

```python
# Check the accuracy in the test dataset
y_pred = gd.predict(X_test)
print("R2 Score:", r2_score(y_test, gd.predict(X_test)))
```

```python
# Plot the regression line
sns.scatterplot(x=X_train.flatten(), y=y_train)
sns.lineplot(x=X_train.flatten(), 
             y=gd.predict(X_train).flatten(), 
             c="#b10026", label="Regression Line")
plt.show()
```

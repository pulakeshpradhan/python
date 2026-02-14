[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/02_Polynomial_Regression/00_Polynomial_Regression.ipynb)

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

## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

## **Make a Data for Polynomial Regression**

```python
# Generate X value
X = 6 * np.random.rand(200, 1) - 3

# y = 0.8X^2 + 0.9x + 2
# Generate the y values
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)
```

```python
# Plot the data
sns.scatterplot(x=X.flatten(), y=y.flatten())
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

## **Apply Linear Regression**

```python
from sklearn.linear_model import LinearRegression
```

```python
# Instantiate a LinearRegressin object
lr = LinearRegression()

# Fit the data
lr.fit(X_train, y_train)
```

```python
# Print intercept and coefficient values
print("Coefficients:", lr.coef_)
print("intercept:", lr.intercept_)
```

```python
# Predict the test data
y_pred = lr.predict(X_test)
```

```python
# Calculate the R2 Score
from sklearn.metrics import r2_score
```

```python
print("R2 Score:", r2_score(y_test, y_pred))
```

```python
# Plot the regression line
sns.scatterplot(x=X.flatten(), y=y.flatten())
sns.lineplot(x=X_test.flatten(), y=lr.predict(X_test).flatten(), c="red", label="Regression Line")
plt.show()
```

## **Apply Polynomial Linear Regression**

```python
from sklearn.preprocessing import PolynomialFeatures
```

```python
# Extract polynomial features
# Degree = 2
# Include bias parameter
poly = PolynomialFeatures(degree=2, include_bias=True)

X_train_transformed = poly.fit_transform(X_train)
X_test_transformed = poly.transform(X_test)
```

```python
# Check the first 5 rows of transformed x_train array
print("Polynomial Features: X^0, X^1, X^2")
X_train_transformed[:5, :]
```

```python
# Instantiate a LinearRegression object for Polynomial Regression
poly_lr = LinearRegression()

# Fit the data
poly_lr.fit(X_train_transformed, y_train)
```

```python
# Print intercept and coefficient values
print("Coefficients:", poly_lr.coef_)
print("intercept:", poly_lr.intercept_)
```

```python
# Predict the test data
y_pred = poly_lr.predict(X_test_transformed)
```

```python
# Calculate the 'R2 Score'
print("R2 Score:", r2_score(y_test, y_pred))
```

```python
# Plot the regression line
sns.scatterplot(x=X.flatten(), y=y.flatten())
sns.lineplot(x=X_test.flatten(), 
             y=poly_lr.predict(X_test_transformed).flatten(), 
             c="red", label="Regression Line")
plt.show()
```

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

```python
# Write a function to see the effect of degree in polynomial regression
def polynomial_regression(degree):
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly.transform(X_new)
    
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
        ("poly_features", polybig_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg)
    ])
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, "r", label="Degree "+str(degree), linewidth=2)
    
    plt.plot(X_train, y_train, "b.", linewidth=3)
    plt.plot(X_test, y_test, "g.", linewidth=3)
    plt.legend(loc="best")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([-3, 3, 0, 10])
    plt.show()
```

```python
polynomial_regression(2)
```

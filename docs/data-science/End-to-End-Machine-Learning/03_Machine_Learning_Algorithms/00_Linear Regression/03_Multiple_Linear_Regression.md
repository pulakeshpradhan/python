[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/00_Linear Regression/03_Multiple_Linear_Regression.ipynb)

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

# **Multiple Linear Regression**
Multiple linear regression is a statistical method used in predictive modeling and data analysis. It extends simple linear regression, which involves modeling the relationship between a dependent variable (also known as the response variable) and a single independent variable (predictor), to cases where there are multiple independent variables. In multiple linear regression, you have more than one predictor variable.


<center><img src="https://miro.medium.com/v2/resize:fit:1400/0*pJsp76_deJvdDean" style="width:60%"></center>


<center><img src="https://aegis4048.github.io/images/featured_images/multiple_linear_regression_and_visualization.png" style="width:60%"></center>


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")
```

## **Create a Data for Regression**

```python
from sklearn.datasets import make_regression
```

```python
# Create the regression data
# X = independent featues
# y = dependent feature
X, y = make_regression(n_samples=100, 
                       n_features=2, 
                       n_informative=2,
                       n_targets=1,
                       noise=50)
```

```python
# Create a dataframe
data_dict = {"feature1": X[:, 0], "feature2": X[:, 1], "target":y}
df = pd.DataFrame(data=data_dict)
print(df.shape)
df.head()
```

## **Plot the Data**

```python
# Plot a 3-dimensional scatter plot
fig = px.scatter_3d(data_frame=df, x="feature1", y="feature2", 
                    z="target", width=600, height=600)
fig.update_traces(marker={'size': 4})
fig.show()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1),
                                                    df["target"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

## **Train a Linear Regression Model**

```python
from sklearn.linear_model import LinearRegression
```

```python
# Instantiate a linear Regression object
lr = LinearRegression()

# Fit the training data
lr.fit(X_train, y_train)
```

```python
# Print the coefficients
print("Coefficients:", lr.coef_)
```

```python
# Print the intercept value
print("Intercept:", lr.intercept_)
```

```python
# Predict the test data
y_pred = lr.predict(X_test)
y_pred
```

## **Accuracy Assessment**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

```python
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

## **Plot the Regression Plane**

```python
# Check the minimum value of the data
df.min()
```

```python
# Check the maximum value of the data
df.max()
```

```python
# Make a mesh grid
x = np.linspace(start=-3, stop=3, num=10)
y = np.linspace(start=-3, stop=3, num=10)
xGrid, yGrid = np.meshgrid(y, x)
```

```python
# Combine x and y cor=ordinates grid
final = np.vstack((xGrid.ravel().reshape(1, 100), yGrid.ravel().reshape(1, 100))).T

# Predict the z value
final_z = lr.predict(final).reshape(10, 10)
z = final_z
```

```python
fig = px.scatter_3d(data_frame=df, x="feature1", y="feature2", 
                    z="target", width=600, height=600)
fig.add_trace(go.Surface(x=x, y=y, z=z))
fig.show()
```

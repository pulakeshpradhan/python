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

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/04_Machine_Learning_Algorithms/03_Multiple_Linear_Regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="5bd4e2da-eff3-46f0-b598-ef42e17255fd" -->
# **Multiple Linear Regression**
Multiple linear regression is a statistical method used in predictive modeling and data analysis. It extends simple linear regression, which involves modeling the relationship between a dependent variable (also known as the response variable) and a single independent variable (predictor), to cases where there are multiple independent variables. In multiple linear regression, you have more than one predictor variable.
<!-- #endregion -->

<!-- #region id="fdb8ea46-591f-48b5-b028-c626e897091f" -->
<center><img src="https://miro.medium.com/v2/resize:fit:1400/0*pJsp76_deJvdDean" width="60%"></center>
<!-- #endregion -->

<!-- #region id="857ff8c8-fd7e-424c-810f-96e8bc5fb813" -->
<center><img src="https://aegis4048.github.io/images/featured_images/multiple_linear_regression_and_visualization.png" width="60%"></center>
<!-- #endregion -->

<!-- #region id="c0a6dc83-bdec-44a9-bd3d-f35d992b53a3" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SmwCeFYMeEg5" outputId="f19d2f71-68af-48df-e11f-64511a6c0a37"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="ce7070f2-9e93-4722-87b1-86362f9c8bcd"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="33a1f01f-1dbc-422e-9821-e9d2d71c5d11" -->
## **Generate a Data for Regression**
<!-- #endregion -->

```python id="8b042bfd-f474-4b4a-882a-682f81e16080"
# Generate a data for regression
# X = independent featues
# y = dependent feature
X, y = make_regression(n_samples=100,
                       n_features=2,
                       n_informative=2,
                       n_targets=1,
                       noise=50)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 223} id="e60f7f09-257f-476f-9661-cb2a1a1ffcf2" outputId="efad2867-007d-40a3-d6b3-0b82a89ee20e"
# Create a dataframe
data_dict = {"feature1": X[:, 0], "feature2": X[:, 1], "target":y}
df = pd.DataFrame(data=data_dict)
print(df.shape)
df.head()
```

<!-- #region id="fe4b08c7-b70f-4ea3-85a4-da8f930c2ac0" -->
## **Plot the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="20b43982-e95d-4912-ab0b-48e07cb10725" outputId="f1cdb3bb-ea58-4cbc-b548-158973c1ee4a"
# Plot a 3-dimensional scatter plot
fig = px.scatter_3d(data_frame=df, x="feature1", y="feature2",
                    z="target", width=600, height=600)
fig.update_traces(marker={'size': 4})
fig.show()
```

<!-- #region id="910fbb6d-fd5e-4e99-916b-d4abe87803e6" -->
## **Train Test Split**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5b45d5b7-60b3-4957-80b9-1d40930f0057" outputId="42e62175-130d-4021-ea06-e6e3d9bb27f4"
X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1),
                                                    df["target"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

<!-- #region id="c34d7e65-75aa-482d-94cb-8dace550ff16" -->
## **Train a Linear Regression Model**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 74} id="5589bca6-c63b-4756-b2ff-6fe7e6f4ce4e" outputId="c8761826-446f-4bc5-81b7-14d8fa8c95c3"
# Instantiate a linear Regression object
lr = LinearRegression()

# Fit the training data
lr.fit(X_train, y_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="d41bfcff-3ce3-446b-9787-1acdd82c1fb3" outputId="a23162cf-ded5-46ef-b8dd-05dacd031ffb"
# Print the coefficients
print("Coefficients:", lr.coef_)
```

```python colab={"base_uri": "https://localhost:8080/"} id="b3bbab38-d22d-421b-8876-b44e0ef2f0ea" outputId="094edf3f-c1b4-4e20-e626-bdfc440b4f8d"
# Print the intercept value
print("Intercept:", lr.intercept_)
```

```python colab={"base_uri": "https://localhost:8080/"} id="cf692a92-d1e0-418b-ad8c-eb120497af9c" outputId="66638a39-10a9-469f-c9d7-b46f835ab5a2"
# Predict the test data
y_pred = lr.predict(X_test)
y_pred
```

<!-- #region id="3ce30ed8-611f-41cc-bcda-a6356cdc9b91" -->
## **Accuracy Assessment**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="404fef22-da23-4ec3-9f7e-60b46a8af64b" outputId="cd69c8ba-a6b9-4be7-eb23-2a711c76a073"
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

<!-- #region id="2b6a4230-f855-4a7c-b5bf-d7dcae582b7f" -->
## **Plot the Regression Plane**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1ec1f958-7584-4a94-af2d-10b58bd4e0a2" outputId="1765c27f-c269-4fb9-f3d6-094d3960b921"
# Check the minimum value of the data
df.min()
```

```python colab={"base_uri": "https://localhost:8080/"} id="b5fbd295-00ae-4e4a-9f25-011dbce3adbc" outputId="5a7aacea-72c6-496e-c11d-368c54f205a1"
# Check the maximum value of the data
df.max()
```

```python id="cb8f49e9-9be0-414f-8c43-565486ff4d48"
# Make a mesh grid
x = np.linspace(start=-3, stop=3, num=10)
y = np.linspace(start=-3, stop=3, num=10)
xGrid, yGrid = np.meshgrid(y, x)
```

```python id="b1752ad3-95fe-4201-a87a-0b1be3f7082b"
# Combine x and y cor=ordinates grid
final = np.vstack((xGrid.ravel().reshape(1, 100), yGrid.ravel().reshape(1, 100))).T

# Predict the z value
final_z = lr.predict(final).reshape(10, 10)
z = final_z
```

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="8540897f-c374-48fb-8419-22a43916e7ba" outputId="628a7e34-babc-488f-afaf-caf9c771563c"
fig = px.scatter_3d(data_frame=df, x="feature1", y="feature2",
                    z="target", width=600, height=600)
fig.update_traces(marker={'size': 4})
fig.add_trace(go.Surface(x=x, y=y, z=z))
fig.show()
```

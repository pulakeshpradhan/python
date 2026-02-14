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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/04_Machine_Learning_Algorithms/00_Simple_Linear_Regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="b00af623-63c3-4d87-8128-47a6f7e6acc3" -->
# **Simple Linear Regression**

Simple linear regression is a statistical method used to model the relationship between two variables, typically denoted as X and Y. It is a straightforward approach to understanding how changes in one variable (X) are associated with changes in another variable (Y). The goal of simple linear regression is to find a linear equation that best represents this relationship.
<!-- #endregion -->

<!-- #region id="ab0c9d74-8705-4053-b6ff-3ee95ec02dc3" -->
<center><img src="https://miro.medium.com/v2/resize:fit:1400/1*GSAcN9G7stUJQbuOhu0HEg.png" width="50%"></center>
<!-- #endregion -->

<!-- #region id="d69bff37-da9d-4619-accf-982eb62a1f75" -->
<center><img src="https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704-EP713_MultivariableMethods/SimpleLinearRegression.png" width="60%"></center>
<!-- #endregion -->

<!-- #region id="b437c0d4-32da-4fb2-a294-1a7b9e58bea6" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wcGCl3-qyif1" outputId="5371da08-6333-404f-c5d4-db4800074b64"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="d38d3d6b-30d7-4845-a298-6223bc76568c"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="989f11ff-1a12-4d83-bb8c-f9f7b567c73f" -->
## **Read the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="76b55681-c21a-4003-b5d5-64f2353a2a4e" outputId="8acc74e8-a16b-463e-c4a6-9101214ded6d"
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/Placement_SLR.csv")
df
```

<!-- #region id="0a04f3dd-91f1-4c11-b6e2-6e9e851ca22f" -->
## **Plot the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="2e07ae22-b215-492d-a111-d552badc3246" outputId="e59bc0e8-242e-49e3-ea1e-c8440f7c759d"
sns.scatterplot(x=df["cgpa"], y=df["package"])
plt.title("Scatterplot between CGPA and Package")
plt.show()
```

<!-- #region id="d73b16f6-db55-4f3e-b87b-f3fc79a9f40b" -->
## **Train Test Split**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="13db0206-bf93-4cd3-9237-a41c0084f9ae" outputId="35e0458c-5504-4d12-ae42-017f731adca3"
X_train, X_test, y_train, y_test = train_test_split(df["cgpa"],
                                                    df["package"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

<!-- #region id="7772fa64-424d-4330-8374-4e41ea5af701" -->
## **Train a Simple Linear Regression Model**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 74} id="dc710d01-3674-46ef-95a0-b283c42abeee" outputId="118d97be-7c8a-4224-ee50-0843296153c3"
# Instantiate a LinearRegression object
lr = LinearRegression()

# Fit the training data
lr.fit(X_train.values.reshape(140, 1), y_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="475dd1be-18df-4c1d-bac1-0354a3e154e1" outputId="e5ba6f17-6821-45fa-bb67-6288faa49e5a"
# Predict the Test data
y_pred = lr.predict(X_test.values.reshape(60, 1))
y_pred
```

<!-- #region id="3f2750d9-622b-4777-9579-c0e7f2e3da4f" -->
## **Plot the Best Fit Line**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="89ca947a-241c-4ea0-b7b4-115d6515667d" outputId="055d4bd6-39eb-4e49-d35d-f8316b2f9dd1"
sns.scatterplot(x=df["cgpa"], y=df["package"])
sns.lineplot(x=X_train, y=lr.predict(X_train.values.reshape(140, 1)), c="red",
             label="Regression Line")
plt.title("Scatterplot between CGPA and Package")
plt.show()
```

<!-- #region id="90186667-4c6b-457a-bde3-e9282e3bf33f" -->
## **Fetch the Slope and Y-intercept Value**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="f2af85e5-08c4-45ef-9b4f-3089e59ddf8a" outputId="b2114144-1d78-40a2-b807-835cfcf215fb"
# Extract the slope value
m = lr.coef_[0]
# Extract the y-intercept value
c = lr.intercept_

print("Slope (m):", m)
print("Y-intercept (c):", c)
```

<!-- #region id="952d63ca-10b0-4b8f-8907-7a7f438f1cd4" -->
## **Check the RMSE**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="33e6b592-9f78-4ace-85f4-f530a0b22404" outputId="31623739-cf4a-4dc3-f580-0aae0ffd6cac"
# Calculate the Root Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error (MSE):", mse.round(2))
print("Root Mean Squared Error (RMSE):", rmse.round(2))
```

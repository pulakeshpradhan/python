[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/00_Linear Regression/01_Mathematical_Formulation_of_SLR.ipynb)

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

# **Mathematical Formulation of Simple Linear Regression**
In simple linear regression, the slope (m) and the intercept (b) of the linear equation can be calculated using the following mathematical formulas:


<center><img src="https://images.contentful.com/piwi0eufbb2g/1RVvEW0TA8UmKfBLuuLoYw/59ec766bb046dec5763333da69549ab2/image.png"></center>

<center><img src="https://miro.medium.com/v2/resize:fit:780/1*o2QwtRXYcmZj7S5k45sbLA.jpeg"></center>

```python
class CustomLR:
    # Constructor
    def __init__(self):
        self.m = None # Slope
        self.b = None # Y-intercept
        
    def fit(self, x_train, y_train):
    
        # Algorithm to find slope (m)
        numerator = 0
        denominator = 0
        for i in range(x_train.shape[0]):
            numerator += ((x_train[i] - x_train.mean()) * (y_train[i] - y_train.mean()))
            denominator += (x_train[i] - x_train.mean()) ** 2
        
        self.m = numerator/denominator
        
        # Algorithm to find y-intercept (b)
        self.b = y_train.mean() - ((numerator/denominator) * x_train.mean())
        
    def predict(self, x_test):
        x_pred = (x_test * self.m) + self.b
        return np.array(x_pred)
```

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

## **Apply Linear Regression with Custom Model**

```python
# Instantiate a Linear Regression Object
lr = CustomLR()

# Fit the training data
lr.fit(x_train.values, y_train.values)
```

```python
# Print the slope and y-intercept value of the LR Model
print("Slope (m):", lr.m)
print("Y-intercept (b):", lr.b)
```

```python
# Predict the test data
y_pred = lr.predict(x_test)
y_pred
```

## **Plot the Best Fit Line**

```python
sns.scatterplot(x=df["cgpa"], y=df["package"])
sns.lineplot(x=x_train, y=lr.predict(x_train), c="red",
             label="Regression Line")
plt.title("Scatterplot between CGPA and Package")
plt.show()
```

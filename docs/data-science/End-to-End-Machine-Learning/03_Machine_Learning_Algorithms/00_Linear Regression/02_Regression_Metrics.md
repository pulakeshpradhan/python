[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/00_Linear Regression/02_Regression_Metrics.ipynb)

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

# **Regression Metrics**
Regression metrics are used to evaluate the performance of predictive models that aim to estimate a continuous target variable. These metrics help assess how well the model's predictions align with the actual values, allowing you to understand the accuracy, precision, and goodness of fit of your regression model. Here are some commonly used regression metrics:


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
df.head()
```

```python
# Check for the null values
df.isnull().sum()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
x_train, x_test, y_train, y_test = train_test_split(df.drop("package", axis=1),
                                                    df["package"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape , x_test.shape
```

## **Plot the Data**

```python
sns.scatterplot(x=x_train["cgpa"], y=y_train)
plt.title("Scatterplot between CGPA and Package")
plt.show()
```

## **Train a Simple Linear Regression Model**

```python
from sklearn.linear_model import LinearRegression
```

```python
# Instantiate an object of the LinearRegression class
lr = LinearRegression()

# Fit the training data
lr.fit(x_train, y_train)
```

```python
# Predict the test data
y_pred = lr.predict(x_test)
y_pred
```

## **Plot the Regression Line**

```python
sns.scatterplot(x=x_train["cgpa"], y=y_train)
sns.lineplot(x=x_test["cgpa"], y=lr.predict(x_test), c="red", label="Regression Line")
plt.title("Scatterplot between CGPA and Package")
plt.show()
```

## **Check the Accuracy using Regression Metrics**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### **Mean Absolute Error (MAE):**
MAE is the average of the absolute differences between the predicted and actual values. It measures the average magnitude of errors and is easy to understand.

**Advantages:**
* Easy to interpret as it represents the average absolute error.
* Resistant to outliers, as it does not square errors.

**Disadvantages:**
* Does not penalize larger errors more heavily.
* May not work well if the error distribution is not symmetric.

**Formula:**<br>
<center><img src="https://editor.analyticsvidhya.com/uploads/42439Screenshot%202021-10-26%20at%209.34.08%20PM.png" style="width:30%"></center>

```python
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae.round(2))
```

### **Mean Squared Error (MSE):**
MSE measures the average of the squared differences between the predicted and actual values. It gives more weight to larger errors and penalizes them.

**Advantages:**
* Provides a measure of how well the model performs while penalizing larger errors.
* Mathematically convenient and commonly used in optimization algorithms.

**Disadvantages:**
* The squared nature of the metric makes it sensitive to outliers.
* The units of MSE are not the same as the target variable, making it less interpretable.

**Formula:**<br>
<center><img src="https://cdn-media-1.freecodecamp.org/images/hmZydSW9YegiMVPWq2JBpOpai3CejzQpGkNG" style="width:30%"> </center>

```python
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse.round(2))
```

### **Root Mean Squared Error (RMSE):**
RMSE is the square root of the MSE. It provides a more interpretable metric in the same units as the target variable.

**Advantages:**
* Shares the same unit as the target variable, which makes it more interpretable than MSE.
* Balances the sensitivity to outliers found in MSE.

**Disadvantages:**
* Like MSE, it can still be sensitive to outliers.
* Not as intuitive as MAE.

**Formula:**<br>
<center><img src="https://miro.medium.com/v2/resize:fit:966/1*lqDsPkfXPGen32Uem1PTNg.png" style="width:25%"> </center>

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse.round(2))
```

### **R-squared (R²):**
R-squared measures the proportion of the variance in the target variable that is explained by the model. It ranges from 0 to 1, with higher values indicating a better fit.

**Advantages:**
* Provides a measure of goodness of fit, indicating how well the model explains the variance in the data.
* Values range from 0 to 1, where higher values suggest a better fit.

**Disadvantages:**
* It may increase when adding more predictors, even if they are irrelevant (overfitting).
* R-squared alone doesn't reveal the direction or magnitude of individual errors.

**Formula:**<br>
<center><img src="https://i0.wp.com/www.fairlynerdy.com/wp-content/uploads/2017/01/r_squared_5.png?resize=625%2C193" style="width:30%"> </center>

<center><img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*EjMnICEPkm0VzoLetqeYbw.jpeg" style="width:60%"> </center>

```python
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2.round(2))
```

### **Adjusted R-squared:**
Adjusted R-squared is a modified version of R-squared that accounts for the number of predictors in the model. It penalizes the addition of irrelevant predictors.

**Advantages:**
* Adjusts R-squared to account for the number of predictors, helping to mitigate overfitting.
* Offers a more realistic assessment of model fit in multiple regression.

**Disadvantages:**
* It can still be influenced by outliers and unrepresentative samples.

**Formula:**<br>
<center><img src="https://i.stack.imgur.com/RcGf6.png" style="width:50%"> </center>

```python
n = len(x_test) # Total Sample Size
p = len(x_test.columns) # Number of independent variable
```

```python
adusted_r2 = 1 - ((1 - r2)*(n - 1) / (n - p - 1))
print("Adjusted R2 Score:", adusted_r2.round(2))
```

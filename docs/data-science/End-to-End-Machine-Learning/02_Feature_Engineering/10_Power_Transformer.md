[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/10_Power_Transformer.ipynb)

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

# **Power Transformer**
Power transforms are a family of parametric, monotonic transformations that are applied to make data more Gaussian-like. This is useful for modeling issues related to heteroscedasticity (non-constant variance), or other situations where normality is desired. Currently, PowerTransformer supports the Box-Cox transform and the Yeo-Johnson transform. The optimal parameter for stabilizing variance and minimizing skewness is estimated through maximum likelihood. Box-Cox requires input data to be strictly positive, while Yeo-Johnson supports both positive or negative data. By default, zero-mean, unit-variance normalization is applied to the transformed data.


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
```

## **Read the Data**

```python
df = pd.read_csv("D:\Coding\Datasets\concrete.csv")
df
```

```python
# Check the null values
df.isnull().sum()
```

```python
# Describe the data
df.describe()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop(["strength"], axis=1),
                                                    df["strength"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

## **Train a Linear Regression Model**

```python
from sklearn.linear_model import LinearRegression
```

```python
# Instantiate a LinearRegression object
lr = LinearRegression()

# Fit the training data
lr.fit(x_train, y_train)
```

## **Accuracy Assessment**

```python
# Predict the test data with the LR model
y_pred = lr.predict(x_test)
```

```python
# Calculate the R2 score
from sklearn.metrics import r2_score

print("R2 Score of Linear Regression Model:", r2_score(y_test, y_pred))
```

```python
# Calculate accuracy after cross validation
from sklearn.model_selection import cross_val_score

# Instantiate a new Linear Regression model
lr = LinearRegression()

# Print the cross validation score
print("R2 Score of Linear Regression model after Cross Validation:", 
      np.mean(cross_val_score(estimator=lr, X=x_train, y=y_train, cv=10, scoring="r2")))
```

## **Data Visualization**

```python
# Plot the histplot of each and every column without any transformation

for column in x_train.columns:
    plt.figure(figsize=(14, 4))
    
    plt.subplot(121)
    sns.histplot(x_train[column], kde=True, bins=30)
    plt.title(column.title() + " Histplot")
    
    plt.subplot(122)
    stats.probplot(x_train[column], dist="norm", plot=plt)
    plt.title(column.title() +" QQ plot")
    
    plt.show()
```

## **Apply Box-Cox Transformation**


The Box-Cox transformation is a mathematical technique used in statistics to stabilize variance and make data approximately follow a normal distribution. It is particularly useful when you are dealing with data that violates the assumptions of normality and constant variance, which are common assumptions in many statistical methods, including linear regression.


**Formula:**<br>
<center><img src="https://miro.medium.com/v2/resize:fit:884/1*_nHvGg-OQYtNSSZph-424g.png" style="height:200px"></center>


Here:
- **y** is the original data.
- **y^(lambda)** is the transformed data.
- **lambda** is the transformation parameter. It can be any real number, but in practice, it is often chosen to maximize the normality of the transformed data. This parameter determines the type of transformation applied to the data.


**Significance:**

1. **Stabilizes Variance**: When **lambda** is not equal to 1, the transformation can be used to stabilize the variance of the data. If the data has varying variances at different levels, the Box-Cox transformation can help make the variances more constant.

2. **Normalizes Distribution**: When **lambda** is chosen appropriately, the transformed data tends to follow a normal distribution more closely. This can be beneficial when using statistical techniques that assume normally distributed data.

3. **Handles Non-Negative Data**: The Box-Cox transformation is typically applied to non-negative data, as it involves taking logarithms and raising to powers. If your data contains negative values, you may need to add a constant to make it non-negative before applying the transformation.


<center><img src="https://media.geeksforgeeks.org/wp-content/uploads/20200531232546/output275.png" style="height:400px"></center>


### **Transform the Data using Box-Cox Transformer**

```python
from sklearn.preprocessing import PowerTransformer
```

```python
# Create an object of the PowerTransformer class
box_cox_transformer = PowerTransformer(method="box-cox", standardize=True)

# Fit and Tranform the training data
# Add a small value because Box-Cox transformation can only be applied to strictly positive data
x_train_transformed = box_cox_transformer.fit_transform(x_train + 0.00000001)

# Transform the testing data
x_test_transformed = box_cox_transformer.transform(x_test + 0.00000001)
```

```python
# Convert the transformed array into pandas dataframe
x_train_transformed = pd.DataFrame(x_train_transformed, columns=x_train.columns)
x_test_transformed = pd.DataFrame(x_test_transformed, columns=x_test.columns)
```

```python
x_train_transformed.head()
```

### **Fetch the value of λ for Each Column**

```python
# Convert the λ values into pandas series
lambda_values_box_cox = pd.Series(data=box_cox_transformer.lambdas_, index=x_train.columns)
lambda_values_box_cox
```

### **Apply Linear Regression on Box-Cox Transformed Data**

```python
# Instantiate a Linear Regression object
lr = LinearRegression()

# Fit the training data
lr.fit(x_train_transformed, y_train)

# Prdict the test data
y_pred = lr.predict(x_test_transformed)

# Calculate R2 Score
print("R2 Score of Linear Regression Model after Box-Cox Transformation:", r2_score(y_test, y_pred))
```

```python
# Check accuracy after cross validation
print("R2 Score of Linear Regression Model after Box-Cox Transformation and Cross Validation:",
      np.mean(cross_val_score(lr, x_train_transformed, y_train, scoring="r2", cv=10)))
```

### **Display the Box-Cox Transformed Data**

```python
for column in x_train_transformed.columns:
    plt.figure(figsize=(14, 4))
    
    plt.subplot(121)
    sns.histplot(x_train[column], kde=True, bins=30)
    plt.title(column.title() + " Histplot")
    
    plt.subplot(122)
    sns.histplot(x_train_transformed[column], kde=True, bins=30)
    plt.title(column.title() + " Histplot after Transformation")
    
    plt.show()
```

## **Apply Yeo-Johnson Transformer**

The Yeo-Johnson transformation is a data transformation technique used in statistics and data analysis. It is primarily used for stabilizing variance and making data more closely follow a normal distribution. This transformation is an extension of the more commonly known Box-Cox transformation and was introduced by Yeo and Johnson in 2000 to address some of its limitations.

The Yeo-Johnson transformation works by applying a mathematical formula to each data point in a given dataset. Unlike the Box-Cox transformation, which is defined only for positive values, the Yeo-Johnson transformation can be applied to both positive and negative values and even zero values.


**Formula:**<br>
<center> <img src="https://graphworkflow.files.wordpress.com/2019/01/yeo_johnson.png" style="height:200px"></center>


Here:<br>
- **y** is the original data point.
- **y^(lambda)** is the transformed value.
- **lambda** is a parameter that determines the transformation. It can take any real value, including zero.


### **Transform the Data using Yeo-Johnson Transformer**

```python
# Create an object of the PowerTransformer class
yeo_johnson_transformer = PowerTransformer(method="yeo-johnson", standardize=True)

# Fit and Tranform the training data
x_train_transformed = yeo_johnson_transformer.fit_transform(x_train)

# Transform the testing data
x_test_transformed = yeo_johnson_transformer.transform(x_test)
```

```python
# Convert the transformed array into pandas dataframe
x_train_transformed = pd.DataFrame(x_train_transformed, columns=x_train.columns)
x_test_transformed = pd.DataFrame(x_test_transformed, columns=x_test.columns)
```

```python
x_train_transformed.head()
```

### **Fetch the value of λ for Each Column**

```python
lambda_values_yeo_johnson = pd.Series(data=yeo_johnson_transformer.lambdas_, index=x_train.columns)
lambda_values_yeo_johnson
```

### **Apply Linear Regression on Box-Cox Transformed Data**

```python
# Instantiate a Linear Regression object
lr = LinearRegression()

# Fit the yeo-johnson transformed training data
lr.fit(x_train_transformed, y_train)

# Predict the test data
y_pred = lr.predict(x_test_transformed)

# Calculate the R2 Score
print("R2 Score of Linear Regression Model after Yeo-Johnson Transformation:", r2_score(y_test, y_pred))
```

```python
# Check accuracy after cross validation
print("R2 Score of Linear Regression Model after Yeo-Johnson Transformation and Cross Validation:",
       np.mean(cross_val_score(lr, x_train_transformed, y_train, scoring="r2", cv=10)))
```

### **Display the Yeo-Johnson Transformed Data**

```python
for column in x_train_transformed.columns:
    plt.figure(figsize=(14, 4))
    
    plt.subplot(121)
    sns.histplot(x_train[column], kde=True, bins=30)
    plt.title(column.title() + " Histplot")
    
    plt.subplot(122)
    sns.histplot(x_train_transformed[column], kde=True, bins=30)
    plt.title(column.title() + " Histplot after Transformation")
    
    plt.show()
```

## **Compare the λ Values between Box-Cox and Yeo-Johnson Transformation** 

```python
# Convert the lambda values into a dataframe
lambda_values = pd.concat((lambda_values_box_cox, lambda_values_yeo_johnson), axis=1)
lambda_values.columns = ["Box_Cox_Lambdas", "Yeo_Johnson_Lambdas"]
```

```python
lambda_values
```

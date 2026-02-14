[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/09_Function_Transformer.ipynb)

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

# **Function Transformer**
FunctionTransformer is a class in scikit-learn that allows you to apply a custom transformation function to your data as part of a scikit-learn pipeline. It is particularly useful when you have a transformation that is not available as a built-in preprocessing step in scikit-learn, and you want to incorporate it into your machine learning workflow.

Here are some key points about FunctionTransformer:

* **Custom Transformation:**<br> FunctionTransformer is used to apply a custom-defined function to transform your data. This function can be any valid Python function that takes an input array-like or pandas DataFrame and returns a transformed version of the data.

* **Seamless Integration:**<br> You can use FunctionTransformer seamlessly within a scikit-learn pipeline, which allows you to create a sequence of data processing and modeling steps. This is helpful for ensuring that your custom transformation is applied consistently to both training and testing data.


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
df = pd.read_csv(r"D:\Coding\Datasets\titanic.csv")
df.head()
```

```python
# Selct only 'Age', 'Fare' and 'Survived' columns
df = df[["Survived", "Age", 'Fare']]
df.head()
```

```python
# Check the column information of the dataframe
df.info()
```

## **Data Preprocessing**

```python
# Fill the null values of the 'Age' column
df["Age"].fillna(df["Age"].mean(), inplace=True)
df.info()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop("Survived", axis=1),
                                                    df['Survived'],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

```python
x_train.head()
```

## **Data Visualization**

```python
# Plot the distribution of the 'Age' column
plt.figure(figsize=(12, 4))

plt.subplot(121)
sns.distplot(x_train["Age"])
plt.title("Age PDF")

plt.subplot(122)
stats.probplot(x_train["Age"], dist="norm", plot=plt)
plt.title("Age QQ Plot")

plt.show()
```

```python
# Plot the distribution of the 'Fare' column
plt.figure(figsize=(12, 4))

plt.subplot(121)
sns.distplot(x_train["Fare"])
plt.title("Fare PDF")

plt.subplot(122)
stats.probplot(x_train["Fare"], dist="norm", plot=plt)
plt.title("Fare QQ Plot")

plt.show()
```

## **Train a Classifier**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
```

```python
# Create object of two different models
lr_clf = LogisticRegression()
dt_clf = DecisionTreeClassifier()
```

```python
# Fit the training data to Logistic Regression model
lr_clf.fit(x_train, y_train)
```

```python
# Fit the training data to Decision Tree model
dt_clf.fit(x_train, y_train)
```

```python
# Predict the test data
y_pred_lr = lr_clf.predict(x_test)
y_pred_dt = dt_clf.predict(x_test)
```

## **Assess the Accuracy**

```python
from sklearn.metrics import accuracy_score
```

```python
print("Accuarcy Score of LR Model:", accuracy_score(y_test, y_pred_lr))
print("Accuracy Score of DT Model:", accuracy_score(y_test, y_pred_dt))
```

## **Apply Transformation**
**Log Transformation:**<br>
Log transformation is a common data preprocessing technique used in various fields such as statistics, data analysis, and machine learning. It involves taking the logarithm of a dataset, typically the natural logarithm (base e) or the base 10 logarithm, to reduce the variation between data points and make the data more suitable for certain analyses or modeling techniques. Log transformation is particularly useful when dealing with data that exhibits exponential or multiplicative growth.

```python
from sklearn.preprocessing import FunctionTransformer
```

```python
# Apply log transformation on 'Fare' column because the PDF is positively/right skewed

# Create an object of FunctionTransformer
log_tranformer = FunctionTransformer(func=np.log1p)
                               
# Fit and transform the 'Fare' column of the training data
x_train_fare = log_tranformer.fit_transform(x_train[["Fare"]])

# Transform the test data
x_test_fare = log_tranformer.transform(x_test[["Fare"]])
```

```python
# Concatenate the columns
x_train_transformed = np.concatenate((x_train[["Age"]], x_train_fare), axis=1)
x_test_transformed = np.concatenate((x_test[["Age"]], x_test_fare), axis=1)
```

```python
# Convert the transformed array into pandas dataframe
x_train_transformed = pd.DataFrame(x_train_transformed, columns=x_train.columns)
x_test_transformed = pd.DataFrame(x_test_transformed, columns=x_train.columns)
```

```python
x_train_transformed.head()
```

```python
# Plot the distribution of 'Fare' column after applying log transformation
plt.figure(figsize=(12, 4))

plt.subplot(121)
sns.distplot(x_train_transformed["Fare"])
plt.title("Fare PDF after Log Transformation")

plt.subplot(122)
stats.probplot(x_train_transformed["Fare"], dist="norm", plot=plt)
plt.title("Fare QQ Plot after Log Transformation")

plt.show()
```

### **Train a Classifier**

```python
# Train the models with log transformed data
lr_clf2 = LogisticRegression()
dt_clf2 = DecisionTreeClassifier()

# Fit the transformed training data
lr_clf2.fit(x_train_transformed, y_train)
dt_clf2.fit(x_train_transformed, y_train)

# Predict the transformed train data
y_pred_lr = lr_clf2.predict(x_test_transformed)
y_pred_dt = dt_clf2.predict(x_test_transformed)
```

### **Assess the Accuracy**

```python
# Assess the Accuracy
print("Accuracy Score of LR Model after log transformation:", accuracy_score(y_test, y_pred_lr))
print("Accuracy Score of DT Model after log transformation:", accuracy_score(y_test, y_pred_dt))
```

### **Apply Cross Validation**

```python
from sklearn.model_selection import cross_val_score
```

```python
# Apply log transformation to 'Age' and 'Fare' columns
x_transformed_2 = log_tranformer.fit_transform(df.drop("Survived", axis=1))
```

```python
lr_clf3 = LogisticRegression()
dt_clf3 = DecisionTreeClassifier()

print("Accuracy Score of LR after Cross Validation:")
print(np.mean(cross_val_score(lr_clf3, x_transformed_2, df["Survived"], scoring="accuracy", cv=10)))

print("Accuracy Score of DT after Cross Validation:")   
print(np.mean(cross_val_score(dt_clf3, x_transformed_2, df["Survived"], scoring="accuracy", cv=10)))
```

## **Create a Function to Apply Different Transformation**

```python
from sklearn.compose import ColumnTransformer
```

```python
def apply_transformation(transformer):
    # Define the training data
    x = df.drop("Survived", axis=1)
    y = df["Survived"]
    
    # Create a function transformer
    transformer = FunctionTransformer(func=transformer)
    
    # Fit the data to the transformer
    x_transformed_fare = transformer.fit_transform(x[["Fare"]])
    
    # Concatenate the columns
    x_transformed = np.concatenate((x[["Age"]], x_transformed_fare), axis=1)
    x_transformed = pd.DataFrame(x_transformed, columns=x.columns)
    
    # Instantiate a logistic regression model
    lr = LogisticRegression()
    
    # Print the accuracy of the model after cross validation
    print("Accuracy Score:")
    print(np.mean(cross_val_score(lr, x_transformed, y, scoring="accuracy", cv=10)))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    stats.probplot(x["Fare"], dist="norm", plot=plt)
    plt.title("Fare QQ Plot Before Transform")
    
    plt.subplot(122)
    stats.probplot(x_transformed["Fare"], dist="norm", plot=plt)
    plt.title("Fare QQ Plot Before Transform")
    
    plt.show()
```

### **Apply Square Transform to 'Fare' Column**

```python
apply_transformation(lambda x:x**2)
```

### **Apply Reciprocal Transform to 'Fare' Column**

```python
apply_transformation(lambda x: 1/(x+0.0000001))
```

### **Apply Square Root Transform to 'Fare' Column**

```python
apply_transformation(lambda x: x**0.5)
```

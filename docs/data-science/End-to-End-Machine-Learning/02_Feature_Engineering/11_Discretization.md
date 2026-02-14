[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/11_Discretization.ipynb)

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

# **Discretization**
Discretization refers to the process of transforming continuous numerical features (variables) into discrete bins or categories. This can be useful in various machine learning tasks, such as decision tree-based algorithms or Naive Bayes, where discrete features are more suitable. Scikit-learn provides a class called **KBinsDiscretizer** for this purpose.


<center><img src="https://www.cradle-cfd.com/dcms_media/image/en_column_basic_fig5.1.jpg" style="width:50%"></center>


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
df = pd.read_csv(r"D:\Coding\Datasets\titanic.csv")
df.head()
```

```python
# Select only necessary columns
df = df[["Age", "Fare", "Survived"]]
df.head()
```

```python
# Check the information of all the columns
df.info()
```

```python
# Drop the null rows
df.dropna(inplace=True)
df.info()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
x_train, x_test, y_train, y_test = train_test_split(df.drop("Survived", axis=1),
                                                    df["Survived"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

```python
x_train.head()
```

## **Train a Classifier**

```python
from sklearn.tree import DecisionTreeClassifier
```

```python
# Instantiate a DecisionTreeClassifier object
dt_clf = DecisionTreeClassifier(random_state=0)

# Fit the training data
dt_clf.fit(x_train, y_train)
```

```python
# Predict the test data
y_pred = dt_clf.predict(x_test)
y_pred
```

## **Check the Accuracy**

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
```

```python
print("Accuracy of Decision Tree Model:", accuracy_score(y_test, y_pred))
```

```python
print("Accuracy of Decision Tree Model after Cross Validation:",
      np.mean(cross_val_score(dt_clf, x_train, y_train, cv=10, scoring="accuracy")))
```

## **Apply Discretization**
In discretization, the "strategy" refers to the method or approach used to determine the bin edges or thresholds when converting continuous numerical features into discrete bins or categories. Scikit-learn's KBinsDiscretizer provides several strategies to choose from:

* **'uniform':** In this strategy, the bins are uniformly spaced across the range of the input data. It divides the range into equal-width intervals. This strategy is simple and can work well when the data distribution is approximately uniform.

* **'quantile':** This strategy divides the data into bins such that each bin contains approximately the same number of data points. It is useful when you want to ensure that each bin has a roughly equal number of samples, even if the data distribution is skewed.

* **'kmeans':** In the 'kmeans' strategy, the bin edges are determined using the k-means clustering algorithm. The number of bins is specified by the n_clusters parameter. This strategy can work well when the data has complex distribution patterns and you want to capture those patterns in the discretization.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
```

```python
# Create a discretization object for "Age" column
kbin_age = KBinsDiscretizer(n_bins=10, strategy="quantile", encode="ordinal")

# Create a discretization object for "Fare" column
kbin_fare = KBinsDiscretizer(n_bins=10, strategy="quantile", encode="ordinal")
```

```python
# Create a ColumnTransformer object for transforming the columns
transformer = ColumnTransformer(transformers=[("trf1", kbin_age, ["Age"]),
                                             ("trf2", kbin_fare, ["Fare"])],
                                remainder="passthrough")
```

```python
# Fit and transform the training data
x_train_transformed = transformer.fit_transform(x_train)

# Transform the testing data
x_test_transformed = transformer.transform(x_test)
```

```python
# Check the name of the transformers
transformer.named_transformers_
```

```python
# Check the first transformer
transformer.named_transformers_["trf1"]
```

```python
# Check the number of bins for the first transformer
transformer.named_transformers_["trf1"].n_bins_
```

```python
# Check the discretization intervals for both the transformers
transformer.named_transformers_["trf1"].bin_edges_
```

```python
transformer.named_transformers_["trf2"].bin_edges_
```

```python
# Convert the transformed array into pandas dataframe
x_train_transformed_df = pd.DataFrame(x_train_transformed, columns=["Age", "Fare"])
x_test_transformed_df = pd.DataFrame(x_test_transformed, columns=["Age", "Fare"])
```

```python
x_train_transformed_df.head()
```

```python
# Create a dataframe to compare the transformed values
output = pd.DataFrame({
    "age": x_train["Age"],
    "age_trf": x_train_transformed[:, 0],
    "fare": x_train["Fare"],
    "fare_trf": x_train_transformed[:, 1]
})
```

```python
output["age_labels"] = pd.cut(x=x_train["Age"], 
                              bins=transformer.named_transformers_["trf1"].bin_edges_[0].tolist())
output["fare_labels"] = pd.cut(x=x_train["Fare"],
                               bins=transformer.named_transformers_["trf2"].bin_edges_[0].tolist())
output
```

### **Train a Classifier with Discretized Data**

```python
dt_clf = DecisionTreeClassifier(random_state=0)

# Fit the transformed training data
dt_clf.fit(x_train_transformed_df, y_train)
```

```python
# Predict the transformed testing data
y_pred = dt_clf.predict(x_test_transformed_df)
y_pred
```

### **Check the Accuracy**

```python
print("Accuracy of Decision Tree Model:", accuracy_score(y_test, y_pred))
```

```python
# Check the accuracy after cross validation
print("Accuracy of Decision Tree Model after Cross Validation:",
      np.mean(cross_val_score(dt_clf, x_train_transformed_df, y_train, cv=10, scoring="accuracy")))
```

## **Create a Function to Plot the Discretization**

```python
def discretize(bins, strategy):
    kbin_age = KBinsDiscretizer(n_bins=bins, strategy=strategy, encode="ordinal")
    kbin_fare = KBinsDiscretizer(n_bins=bins, strategy=strategy, encode="ordinal")
    
    transformer = ColumnTransformer(transformers=[("trf1", kbin_age, [0]),
                                                  ("trf2", kbin_fare, [1])],
                                    remainder="passthrough")
    
    x_transformed = transformer.fit_transform(x_train)
    
    plt.figure(figsize=(14, 4))
    
    plt.subplot(121)
    sns.histplot(x_train["Age"], bins=bins)
    plt.title(f"Age Before Discretization ({strategy})")
    
    plt.subplot(122)
    sns.histplot(x_train_transformed[:, 0])
    plt.title(f"Age After Discretization ({strategy})")
    
    plt.show()
    
    plt.figure(figsize=(14, 4))
    
    plt.subplot(121)
    sns.histplot(x_train["Fare"], bins=bins)
    plt.title(f"Fare Before Discretization ({strategy})")
    
    plt.subplot(122)
    sns.histplot(x_train_transformed[:, 1])
    plt.title(f"Fare After Discretization ({strategy})")
    plt.show()
```

```python
# Display the tranformation after applying 'quantile' strategy
discretize(10, "quantile")
```

```python
# Display the tranformation after applying 'uniform' strategy
discretize(10, "uniform")
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/AdaBoost/00_AdaBoost_Step_by_Step.ipynb)

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

# **AdaBoost - Step by Step**


## **Import Required Libraries**

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

## **Create a DataFrame**

```python
# Create a custom dataframe
data = {
    "x1": [1, 2, 3, 4, 5, 6, 6, 7, 9, 9],
    "x2": [5, 3, 6, 8, 1, 9, 5, 8, 9, 2],
    "label": [1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
}
df = pd.DataFrame(data)
df
```

```python
# Plot the data
sns.scatterplot(x=df["x1"], y=df["x2"], hue=df["label"]);
```

## **Step-1: Initialize Weights**
Start with your training dataset. Each example in the dataset is initially given the same weight. This means that every example is equally important at the beginning.

```python
# Assign weights to each rows
# Initial weights = 1/n
df["weights"] = 1 / df.shape[0]
df
```

## **Step-2: Train the First Decision Stump**

```python
X = df.iloc[:, :2].values
y = df.iloc[:, 2].values
```

```python
# Train a decision stump / weak classifier
dt1 = DecisionTreeClassifier(max_depth=1)
dt1.fit(X, y)
```

```python
# Plot the tree
plot_tree(dt1);
```

```python
# Plot the decision region
plot_decision_regions(X, y, clf=dt1, legend=2);
```

```python
# Calculate prediction of the first model
df["y_pred"] = dt1.predict(X)
df
```

## **Step-3: Calculate Model Weight**

```python
# Write a function to calculate the model weight (alpha)
def calculate_model_weights(data):
    df = data
    df["weights"] = 1 / df.shape[0]
    
    error = 0

    for index, row in df.iterrows():
        if row["label"] != row["y_pred"]:
            error += row["weights"]

    alpha = 0.5 * np.log((1 - error) / (error + 0.000000001))

    return alpha
```

```python
# Calculate weight of the first model
alpha1 = calculate_model_weights(df)
alpha1
```

## **Step-4: Update the Row Weights**

```python
# Write a function to update the weights of each row
def update_row_weights(row, alpha):
    if row["label"] == row["y_pred"]:
        return row["weights"] * np.exp(-alpha)

    else:
        return row["weights"] * np.exp(alpha)
```

```python
# Update the weights
df["updated_weights"] = df.apply(lambda row: update_row_weights(row, alpha1), axis=1)
df
```

```python
# Normalized the weights
df["normalized_weights"] = df["updated_weights"] / df["updated_weights"].sum()
df
```

```python
df["normalized_weights"].sum()
```

## **Step-5: Initialize Ranges**

```python
def initialize_ranges(data):
    range_data = pd.DataFrame(data)
    range_data["cumsum_upper"] = np.cumsum(range_data["normalized_weights"])
    range_data["cumsum_lower"] = (
        range_data["cumsum_upper"] - range_data["normalized_weights"]
    )
    range_data = range_data[
        [
            "x1",
            "x2",
            "label",
            "weights",
            "y_pred",
            "updated_weights",
            "normalized_weights",
            "cumsum_lower",
            "cumsum_upper",
        ]
    ]
    return range_data
```

```python
df = initialize_ranges(df)
df
```

## **Step-6: Create a New Dataset for Upsampling**

```python
# Write a function to create a new dataset based on ranges
def create_new_dataset(data):
    indices = []

    for i in range(data.shape[0]):
        a = np.random.random()
        for index, row in data.iterrows():
            if row["cumsum_lower"] < a < row["cumsum_upper"]:
                indices.append(index)

    print(indices)
    new_data = data.iloc[indices, [0, 1, 2, 3]]

    return new_data
```

```python
df2 = create_new_dataset(df)
df2
```

## **Steps-7: Iterate the Process for All the Decision Stumps**


### **Second Decision Stumps**

```python
X = df2.iloc[:, :2].values
y = df2.iloc[:, 2].values
```

```python
# Initialize the second decision stump
dt2 = DecisionTreeClassifier(max_depth=1)
dt2.fit(X, y)
```

```python
# Plot the tree
plot_tree(dt2);
```

```python
# Plot the decision region
plot_decision_regions(X, y, clf=dt2, legend=2);
```

```python
# Calculate prediction of the second model
df2["y_pred"] = dt2.predict(X)
df2
```

```python
# Calculate weight(alpha) for the second model
alpha2 = calculate_model_weights(df2)
alpha2
```

```python
# Update the row weights
df2["updated_weights"] = df2.apply(lambda row: update_row_weights(row, alpha2), axis=1)
df2
```

```python
# Normalized the weights
df2["normalized_weights"] = df2["updated_weights"] / df2["updated_weights"].sum()
df2
```

```python
# Initialize ranges
df2 = initialize_ranges(df2)
df2
```

```python
# Create a new dataframe
df3 = create_new_dataset(df2)
df3
```

### **Third Decision Stumps**

```python
X = df3.iloc[:, :2].values
y = df3.iloc[:, 2].values
```

```python
# Initialize the third decision stump
dt3 = DecisionTreeClassifier(max_depth=1)
dt3.fit(X, y)
```

```python
# Plot the tree
plot_tree(dt3);
```

```python
# Plot the decision region
plot_decision_regions(X, y, clf=dt3, legend=2);
```

```python
# Calculate prediction of the third model
df3["y_pred"] = dt3.predict(X)
df3
```

```python
# Calculate weight(alpha) for the third model
alpha3 = calculate_model_weights(df3)
alpha3
```

```python
print("alpha1:", alpha1)
print("alpha2:", alpha2)
print("alpha3:", alpha3)
```

## **Step-8: Calculate the Output**

```python
# Select one query points
query1 = df.iloc[0, :2].values.reshape(1, 2)
query1
```

```python
# Predict the query with three decision stumps
pred1 = (
    (alpha1 * dt1.predict(query1)[0])
    + (alpha2 * dt2.predict(query1)[0])
    + (alpha3 * dt3.predict(query1))[0]
)
pred1 = np.sign(pred1)
pred1
```

```python
# Select another query points
query2 = df.iloc[8, :2].values.reshape(1, 2)
query2
```

```python
# Predict the query with three decision stumps
pred2 = (
    (alpha1 * dt1.predict(query2)[0])
    + (alpha2 * dt2.predict(query2)[0])
    + (alpha3 * dt3.predict(query2))[0]
)
pred2 = np.sign(pred2)
pred2
```

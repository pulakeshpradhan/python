[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/05_Decision_Tree/01_Regression_Tree.ipynb)

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

# **Regression Trees**


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

## **Load the Data**

```python
# Read the California housing data
df = pd.read_csv("D:\Coding\Datasets\housing.csv")
print(df.shape)
df.head()
```

## **Data Preprocessing**

```python
# Apply one hot encoding on 'ocean_proximity' column
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
df.head()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test, y_train, y_test = train_test_split(df.drop("median_house_value", axis=1),
                                                    df["median_house_value"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

## **Train a Decision Tree Regression Model**

```python
from sklearn.tree import DecisionTreeRegressor
```

```python
# Instantiate a decision tree regressor object
dtr = DecisionTreeRegressor()

# Fit the training data
dtr.fit(X_train, y_train)
```

```python
# Predict the test data
y_pred = dtr.predict(X_test)
```

```python
# Check the cross validation score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
```

```python
print("Cross Validation R2 Score:", 
      np.mean(cross_val_score(dtr, X_train, y_train, scoring="r2", cv=10)))
```

## **Hyperparameter Tuning**

```python
from sklearn.model_selection import GridSearchCV
```

```python
# Instantiate another decision tree regressor object
dtr2 = DecisionTreeRegressor()
```

```python
# Create the parameters grid
dtr2_param_grid = {
    "max_depth": [2, 4, 8, 10, None],
    "criterion": ["absolute_error", "squared_error"],
    "min_samples_split": [4, 8, 10, 12, None],
    "min_samples_leaf": [1, 2, 4, 6, None],
    "max_features": [0.25, 0.5, 0.75, 1.0, None]
}
```

```python
# Apply the GridSearchCV to find the best hyperparameters
dtr2_grid = GridSearchCV(estimator=dtr2,
                         param_grid=dtr2_param_grid,
                         scoring="r2",
                         n_jobs=-1,
                         cv=5,
                         verbose=2)

# Fit the training data
dtr2_grid.fit(X_train, y_train)
```

```python
# Print the best parameters
dtr2_grid.best_params_
```

```python
# Print the best score
dtr2_grid.best_score_
```

```python
# Train the model with best parameters
dtr2 = dtr2_grid.best_estimator_
dtr2
```

## **Accuracy Assessment**

```python
# Predict the test data
y_pred = dtr2.predict(X_test)
```

```python
# Calculate the R2 Score
print("R2 Score:", r2_score(y_test, y_pred))
```

## **Feature Importance**

```python
# Extract the feature importance in a dataframe
feature_importance = pd.Series(dtr2.feature_importances_, index=X_train.columns)\
                       .sort_values(ascending=False)
feature_importance
```

```python
# Plot the feature importance
plt.figure(figsize=(4, 5))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance")
plt.show()
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/Flood-Susceptibility-Zonation-of-Malda-District/Notebooks/5_Application_of_Random_Forest_for_Flood_Susceptibility_Zonation.ipynb)

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

# **Application of Random Forest for Flood Susceptibility Zonation**


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
```

## **Read the Data**

```python
training_df = pd.read_csv("D:\Research Works\Flood\Flood_Risk_Zonation_of_Maldah\Datasets\CSVs\Training_Data.csv")
testing_df = pd.read_csv("D:\Research Works\Flood\Flood_Risk_Zonation_of_Maldah\Datasets\CSVs\Testing_Data.csv")
```

```python
training_df.head()
```

```python
# Drop the geometry columns
training_df.drop("geometry", axis=1, inplace=True)
testing_df.drop("geometry", axis=1, inplace=True)

training_df.columns == testing_df.columns
```

```python
training_df.shape
```

```python
# Check for the null values
training_df.isnull().sum().sum()
```

```python
testing_df.isnull().sum().sum()
```

## **Specify the Train Test Data**

```python
X_train, y_train = training_df.drop("Flood", axis=1), training_df["Flood"]
X_test, y_test = testing_df.drop("Flood", axis=1), testing_df["Flood"]

X_train.shape, X_test.shape
```

```python
X_train.head()
```

```python
# Change the datatype into float
X_train = X_train.astype(float)
X_test = X_test.astype(float)
```

## **Feature Selection using Information Gain**

```python
from sklearn.feature_selection import mutual_info_classif
```

```python
# Determine the mutual information
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info
```

```python
# Convert the mutual info array into a pandas series
mutual_info = pd.Series(mutual_info, index=X_train.columns).sort_values(ascending=False)
mutual_info
```

```python
# Plot the orderd mutual_info values per feature
plt.figure(figsize=(12, 4), dpi=100)

# Define a color palette
color_palette = sns.color_palette(palette="coolwarm", n_colors=len(mutual_info))

sns.barplot(x=mutual_info.index, y=mutual_info, palette=color_palette,
            edgecolor="black", linewidth=0.5)
plt.title("Mutual Information", fontname="Times New Roman", color="black", fontsize=12)
plt.xticks(rotation=90)
plt.xticks(fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
plt.ylabel("Information Gain Value", fontname="Times New Roman")
plt.show()
```

```python
from sklearn.feature_selection import SelectKBest
```

```python
# Select the top 20 important features
selected_features = SelectKBest(mutual_info_classif, k=20)
selected_features.fit(X_train, y_train)
selected_features = X_train.columns[selected_features.get_support()]
selected_features
```

```python
X_train = X_train[selected_features]
X_train
```

```python
X_test = X_test[selected_features]
X_test
```

## **Apply Random Forest Classification**


### **Build a Random Forest Classification Model**

```python
from sklearn.ensemble import RandomForestClassifier
```

```python
# Instantiate a RandomForestClassifier object
rf = RandomForestClassifier()
```

### **Hyperparameter Tuning**

```python
# Define all the hyperparameters

# Number of trees in random forest
n_estimators = [50, 75, 100, 150, 200, 300]

# Criterion
criterion = ["gini", "entropy"]

# Maximum depth of each tree
max_depth = [2, 4, 6, 8, 10, None]

# Number of features to consider at each split
max_features = [0.2, 0.4, 0.6, 0.8, 1.0]

# Minimum number of samples required to split an internal node
min_samples_split = [2, 4, 8, 10]

# Minimumn number of samples required to be a leaf node
min_samples_leaf = [1, 2, 4, 8]

# Number of samples
max_samples = [0.25, 0.5, 0.75, 1.0]

# Bootstrap
bootstrap = [True, False]
```

```python
# Define the parameter grid in a dictionary
rf_param_grid = {"n_estimators": n_estimators,
                 "criterion": criterion,
                 "max_depth": max_depth,
                 "max_features": max_features,
                 "min_samples_split": min_samples_split,
                 "min_samples_leaf": min_samples_leaf,
                 "max_samples": max_samples,
                 "bootstrap": bootstrap}
rf_param_grid
```

```python
from sklearn.model_selection import RandomizedSearchCV
```

```python
# Apply Randomized Search CV
rf_grid = RandomizedSearchCV(estimator=rf,
                             param_distributions=rf_param_grid,
                             n_iter=1000,
                             scoring="accuracy",
                             n_jobs=-1,
                             cv=5,
                             verbose=1)
```

```python
# Fit the training data to GridSearcCV
rf_grid.fit(X_train, y_train)
```

```python
rf_grid.best_params_
```

```python
# Check the best score
rf_grid.best_score_
```

```python
# Build a Random Forest Model with best estimators
rf_final = rf_grid.best_estimator_
rf_final
```

## **Accuracy Assessment**

```python
# Predict the test data
y_pred = rf_final.predict(X_test)
```

```python
from sklearn.metrics import accuracy_score, classification_report
```

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
```

```python
print(classification_report(y_test, y_pred))
```

## **Feature Importance**

```python
# Get the feature importance
feature_importance = rf_final.feature_importances_

# Convert the feature importance into a pandas series
feature_importance = pd.Series(feature_importance, index=X_train.columns)

# Sort the values in descending order
feature_importance = feature_importance.sort_values(ascending=False)
feature_importance
```

```python
# Plot the feature importance
plt.figure(figsize=(6, 8), dpi=100)

# Define a color palette
color_palette = sns.color_palette(palette="coolwarm", n_colors=len(feature_importance))

sns.barplot(x=feature_importance, y=feature_importance.index, palette=color_palette, 
            edgecolor="black", linewidth=0.5)
plt.title("Feature Importance", fontname="Times New Roman", color="black", fontsize=12)
plt.xticks(fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
plt.show()
```

## **Export the Model**

```python
import pickle
```

```python
output_folder = "D:\\Coding\\Git Repository\\Research Repo\\Flood\\Flood-Susceptibility-Zonation-of-Maldah\\Model\\"
model_name = "rf_model.pkl"
```

```python
# Export the model
# pickle.dump(rf_final, file=open(output_folder+model_name, "wb"))
```

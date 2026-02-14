[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/Flood-Susceptibility-Zonation-of-Malda-District/Notebooks/6_Application_of_Xgboost_for_Flood_Susceptibility_Zonation.ipynb)

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

# **Application of Xgboost for Flood Susceptibility Zonation**


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
# Select the important features
selected_features = ['Dist_to_River', 'TWI', 'Rainfall', 'Clay_Content', 'TRI', 'NDVI',
                     'MFI', 'Elevation', 'MNDWI', 'Drainage_Density', 'Geomorphology_Active_Flood_Plain',
                     'Geomorphology_Older_Alluvial_Plain', 'Geomorphology_Older_Flood_Plain',
                     'Lithology_Cl_wi_S_Si_Ir_N', 'Lithology_Fe_Ox_S_Si_Cl',
                     'Lithology_S_Si_Cl', 'Lithology_S_Si_Cl_wi_Cal_Co',
                     'LULC_Agricultural_Field', 'LULC_Built_UP_Area',
                     'LULC_Natural_Vegetation']
```

```python
X_train = X_train[selected_features]
X_train
```

```python
X_test = X_test[selected_features]
X_test
```

## **Apply Xgboost Classification**


### **Build an Xgboost Model**

```python
from xgboost import XGBClassifier
```

```python
# Instantiate a XGBClassifier object
xgb = XGBClassifier()
```

### **Hyperparameter Tuning**

```python
# Define all the hyperparameters for Xgboost Model

# Number of boosting rounds
n_estimators = [25, 50, 100, 150, 200, 300]

# Step size shrinkage
learning_rate = [0.05, 0.1, 0.15, 0.2, 0.3]

# Maximum depth of the trees
max_depth = [2, 4, 6, 8, None]

# Minimum sum of instance weight (hessian) needed in a child
min_child_weight = [1, 3, 5, 7]

# Minimum loss reduction required to make a further partition on a leaf node
gamma = [0.0, 0.1, 0.2, 0.3, 0.4]

# Fraction of features used for fitting the trees
colsample_bytree = [0.3, 0.5, 0.7, 1.0]

# Subsample ratio of the training instance
subsample = [0.3, 0.5, 0.7, 1]
```

```python
# Define the parameter grid in a dictionary
xgb_param_grid = {"n_estimators": n_estimators,
                  "learning_rate": learning_rate,
                  "max_depth": max_depth,
                  "min_child_weight": min_child_weight,
                  "gamma": gamma,
                  "colsample_bytree": colsample_bytree,
                  "subsample": subsample}
xgb_param_grid
```

```python
from sklearn.model_selection import RandomizedSearchCV
```

```python
# Apply Randomized Search CV
xgb_grid = RandomizedSearchCV(estimator=xgb,
                              param_distributions=xgb_param_grid,
                              n_iter=1000,
                              scoring="accuracy",
                              n_jobs=-1,
                              cv=5,
                              verbose=1)
```

```python
# Fit the training data to Randomized Search CV
xgb_grid.fit(X_train, y_train)
```

```python
xgb_grid.best_params_
```

```python
# Check the best score
xgb_grid.best_score_
```

```python
# Build a Xgboost Model with best estimators
Xgb_final = xgb_grid.best_estimator_
Xgb_final
```

## **Accuracy Assessment**

```python
# Predict the test data
y_pred = Xgb_final.predict(X_test)
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
feature_importance = Xgb_final.feature_importances_

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
model_name = "xgb_model.pkl"
```

```python
# Export the model
# pickle.dump(Xgb_final, file=open(output_folder+model_name, "wb"))
```

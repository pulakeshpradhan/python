[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/Flood-Susceptibility-Zonation-of-Malda-District/Notebooks/7_Application_of_Logistic_Regression_for_Flood_Susceptibility_Zonation.ipynb)

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

# **Application of Logistic Regression for Flood Susceptibility Zonation**


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

```python jupyter={"source_hidden": true}
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

## **Apply Logistic Regression Classification**


### **Build a Logistic Regression Model**

```python
from sklearn.linear_model import LogisticRegression
```

```python
# Instantiate a LogisticRegression object
log_reg = LogisticRegression()
```

### **Hyperparameter Tuning**

```python
# Define all the hyperparameters for Logistic Regression Model
log_reg_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
    'max_iter': [100, 200, 300],
    'class_weight': ['balanced', None]
}
```

```python
from sklearn.model_selection import RandomizedSearchCV
```

```python
# Apply Randomized Search CV
log_reg_grid = RandomizedSearchCV(estimator=log_reg,
                                  param_distributions=log_reg_param_grid,
                                  n_iter=1000,
                                  scoring="accuracy",
                                  n_jobs=-1,
                                  cv=5,
                                  verbose=1)
```

```python
# Fit the training data to Randomized Search CV
log_reg_grid.fit(X_train, y_train)
```

```python
log_reg_grid.best_params_
```

```python
# Check the best score
log_reg_grid.best_score_
```

```python
# Build a Logistic Regression Model with best estimators
log_reg_final = log_reg_grid.best_estimator_
log_reg_final
```

## **Accuracy Assessment**

```python
# Predict the test data
y_pred = log_reg_final.predict(X_test)
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
feature_importance = log_reg_final.coef_[0]

# Convert the feature importance into a pandas series
feature_importance = pd.Series(feature_importance, index=X_train.columns)

# Sort the values in descending order
feature_importance = feature_importance.sort_values(ascending=False)
feature_importance
```

```python
# Plot the feature importance
plt.figure(figsize=(8, 4), dpi=100)

# Define a color palette
color_palette = sns.color_palette(palette="coolwarm", n_colors=len(feature_importance))

sns.barplot(x=feature_importance.index, y=feature_importance, palette=color_palette, 
            edgecolor="black", linewidth=0.5)
plt.title("Feature Importance", fontname="Times New Roman", color="black", fontsize=12)
plt.xticks(fontname="Times New Roman", rotation=90)
plt.yticks(fontname="Times New Roman")
plt.show()
```

## **Export the Model**

```python
import pickle
```

```python
output_folder = "D:\\Coding\\Git Repository\\Research Repo\\Flood\\Flood-Susceptibility-Zonation-of-Maldah\\Model\\"
model_name = "log_reg_model.pkl"
```

```python
# Export the model
# pickle.dump(log_reg_final, file=open(output_folder+model_name, "wb"))
```

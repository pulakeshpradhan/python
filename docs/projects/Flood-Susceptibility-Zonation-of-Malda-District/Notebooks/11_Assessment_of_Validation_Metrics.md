[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/Flood-Susceptibility-Zonation-of-Malda-District/Notebooks/11_Assessment_of_Validation_Metrics.ipynb)

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

# **Assessment of Validation Metrics**


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
import pickle
import warnings
warnings.filterwarnings("ignore")
```

## **Load all the Models**

```python
# Import the models
# Import the models
rf_model_path = r"D:\Coding\Git Repository\Research Repo\Flood\Flood-Susceptibility-Zonation-of-Maldah\Model\rf_model.pkl"
xgb_model_path = r"D:\Coding\Git Repository\Research Repo\Flood\Flood-Susceptibility-Zonation-of-Maldah\Model\xgb_model.pkl"
log_reg_model_path = r"D:\Coding\Git Repository\Research Repo\Flood\Flood-Susceptibility-Zonation-of-Maldah\Model\log_reg_model.pkl"
svm_model_path = r"D:\Coding\Git Repository\Research Repo\Flood\Flood-Susceptibility-Zonation-of-Maldah\Model\svm_model.pkl"

rf_model = pickle.load(open(rf_model_path, "rb"))
xgb_model = pickle.load(open(xgb_model_path, "rb"))
log_reg_model = pickle.load(open(log_reg_model_path, "rb"))
svm_model = pickle.load(open(svm_model_path, "rb"))
```

## **Import Test Data**

```python
# Import the testing data
testing_df = pd.read_csv("D:\Research Works\Flood\Flood_Risk_Zonation_of_Maldah\Datasets\CSVs\Testing_Data.csv")
testing_df.head()
```

```python
# Select the best feature
X_test = testing_df[rf_model.feature_names_in_]
y_test = testing_df["Flood"]
```

```python
X_test
```

## **Plot the ROC and AUC**

```python
from sklearn.metrics import roc_curve, roc_auc_score
```

```python
# Store the classifiers in a list
classifiers = [
    ("Logistic Regression", log_reg_model),
    ("Support Vector Machine", svm_model),
    ("Random Forest", rf_model),
    ("XGBoost", xgb_model)
]
```

```python
# Create a plot for ROC curves
plt.figure(figsize=(6, 6), dpi=100)

sns.set(style="whitegrid")
# sns.set(font='Times New Roman')

# Loop through each classifier
for name, classifier in classifiers:
    # Predict the test data
    classifier.predict(X_test)
    
    # Predict probabilities
    y_pred_prob = classifier.predict_proba(X_test)[:, 1]
    
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    
    # Calculate the AUC score
    auc = roc_auc_score(y_test, y_pred_prob)
    
    # Plot the ROC curve
    sns.lineplot(x=fpr, y=tpr, label=f"{name} (AUC = {auc:.3f})")

# Add labels and legend
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate', fontname="Times New Roman")
plt.ylabel('True Positive Rate', fontname="Times New Roman")
plt.xticks(fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
plt.title('Receiver Operating Characteristic (ROC) Curve', fontname="Times New Roman")
legend = plt.legend(loc='best')

for text in legend.get_texts():
    text.set_fontname('Times New Roman')
    text.set_fontsize(12)

# Show the plot
plt.show()
```

## **Plot the Validation Metrics**

```python
from sklearn.metrics import classification_report
```

```python
for name, classifier in classifiers:
    print(name)
    print(classification_report(y_test, classifier.predict(X_test)))
```

## **Plot SHAP Values**

```python
import shap
```

```python
order = X_test.columns
col2num = {col: i for i, col in enumerate(X_test.columns)}

order = list(map(col2num.get, order))
```

```python
order
```

### **Shap Values of Logistic Regression**

```python
# Fits the explainer
log_reg_explainer = shap.Explainer(log_reg_model.predict, X_test)

# Calculates the SHAP values - It takes some time
log_shap_values = log_reg_explainer(X_test)
```

```python
plt.figure()
shap.plots.beeswarm(log_shap_values, max_display=None, order=order, plot_size=(6, 10))
```

### **SHAP Values of Support Vector Machine** 

```python
# Fits the explainer
svm_explainer = shap.Explainer(svm_model.predict, X_test)

# Calculates the SHAP values - It takes some time
svm_shap_values = svm_explainer(X_test)
```

```python
plt.figure()
shap.plots.beeswarm(svm_shap_values, max_display=None, order=order, plot_size=(6, 10))
```

### **SHAP Values of Random Forest**

```python
# Fits the explainer
rf_explainer = shap.Explainer(rf_model.predict, X_test)

# Calculates the SHAP values - It takes some time
rf_shap_values = rf_explainer(X_test)
```

```python
plt.figure()
shap.plots.beeswarm(rf_shap_values, max_display=None, order=order, plot_size=(6, 10))
```

### **SHAP Values of XGBoost**

```python
# Fits the explainer
xgb_explainer = shap.Explainer(xgb_model.predict, X_test)

# Calculates the SHAP values - It takes some time
xgb_shap_values = xgb_explainer(X_test)
```

```python
plt.figure()
shap.plots.beeswarm(xgb_shap_values, max_display=None, order=order, plot_size=(6, 10))
```

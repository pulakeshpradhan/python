[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/projects/Flood-Susceptibility-Zonation-of-Malda-District/Notebooks/10_Classify_the_Image.ipynb)

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

# **Classify the Image**


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import warnings
warnings.filterwarnings("ignore")
```

## **Load the Original Parameters Image**

```python
parameter_img = rasterio.open("D:\Research Works\Flood\Flood_Risk_Zonation_of_Maldah\Datasets\Rasters\Maldah_Flood_Parameters.tif")
```

```python
# Store the image parameters in separate variables
bandNum = parameter_img.count
height = parameter_img.height
width = parameter_img.width
crs = parameter_img.crs
transform = parameter_img.transform
shape = (height, width)
```

## **Read the Image as DataFrame**

```python
image = pd.read_csv("D:\Research Works\Flood\Flood_Risk_Zonation_of_Maldah\Datasets\CSVs\Image_CSV.csv")
image
```

```python
# Remove unnecessary column
image.drop("Unnamed: 0", axis=1, inplace=True)
```

## **Load the Models**

```python
import pickle
```

```python
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

```python
rf_model
```

```python
xgb_model
```

```python
log_reg_model
```

```python
svm_model
```

```python
rf_model.feature_names_in_
```

## **Classify the Image**


### **Classify the Image with Random Forest**

```python
# Classify the image with RF Model
rf_predict = rf_model.predict(image)
```

```python
# Predict the Probability of Classification 
rf_predict_prob = rf_model.predict_proba(image)
rf_predict_prob = rf_predict_prob[:, 1]
rf_predict_prob
```

```python
# Reshape the array
rf_classified_image = rf_predict_prob.reshape((3267, 2351))
rf_classified_image
```

```python
# Plot the image
plt.figure(figsize=(8, 6))
plt.imshow(rf_classified_image, cmap="coolwarm")
plt.title("RF Classified Flood Susceptibility Map")
plt.colorbar(label="Flood Probability")
plt.show()
```

### **Classify the Image with Xgboost**

```python
# Classify the image with Xgboost Model
xgb_predict = xgb_model.predict(image)
```

```python
# Predict the Probability of Classification 
xgb_predict_prob = xgb_model.predict_proba(image)
xgb_predict_prob = xgb_predict_prob[:, 1]
xgb_predict_prob
```

```python
# Reshape the array
xgb_classified_image = xgb_predict_prob.reshape((3267, 2351))
xgb_classified_image
```

```python
# Plot the image
plt.figure(figsize=(8, 6))
plt.imshow(xgb_classified_image, cmap="coolwarm")
plt.title("Xgboost Classified Flood Susceptibility Map")
plt.colorbar(label="Flood Probability")
plt.show()
```

### **Classify the Image with Logistic Regression**

```python
# Classify the image with Logistic Regression Model
log_reg_predict = log_reg_model.predict(image)
```

```python
# Predict the Probability of Classification 
log_reg_predict_prob = log_reg_model.predict_proba(image)
log_reg_predict_prob = log_reg_predict_prob[:, 1]
log_reg_predict_prob
```

```python
# Reshape the array
log_reg_classified_image = log_reg_predict_prob.reshape((3267, 2351))
log_reg_classified_image
```

```python
# Plot the image
plt.figure(figsize=(8, 6))
plt.imshow(log_reg_classified_image, cmap="coolwarm")
plt.title("Logistic Regression Classified Flood Susceptibility Map")
plt.colorbar(label="Flood Probability")
plt.show()
```

### **Classify the Image with Support Vector Machine**

```python
# Classify the image with SVM Model
svm_predict = svm_model.predict(image)
```

```python
# Predict the Probability of Classification 
svm_predict_prob = svm_model.predict_proba(image)
svm_predict_prob = svm_predict_prob[:, 1]
svm_predict_prob
```

```python
# Reshape the array
svm_classified_image = svm_predict_prob.reshape((3267, 2351))
svm_classified_image
```

```python
# Plot the image
plt.figure(figsize=(8, 6))
plt.imshow(svm_classified_image, cmap="coolwarm")
plt.title("SVM Classified Flood Susceptibility Map")
plt.colorbar(label="Flood Probability")
plt.show()
```

## **Export the Images**

```python
# # Save the image to the file
# folder_path = r"D:\Research Works\Flood\Flood_Risk_Zonation_of_Maldah\Datasets\Rasters\Outputs"
# file_name = "\SVM_Flood_Prob.tif"
# location = folder_path + file_name

# output = rasterio.open(
#     location,
#     mode='w',
#     driver="GTiff",
#     width=parameter_img.shape[1],
#     height=parameter_img.shape[0],
#     count=1,
#     crs=crs,
#     transform=transform,
#     dtype=str(svm_classified_image.dtype),
# )

# output.write(svm_classified_image, 1)
# output.close()
```

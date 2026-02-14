[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Image-Analysis-in-Remote-Sensing-with-Python/Machine Learning with Scikit Learn/Satellite_Image_Classification_using_Scikit_Learn.ipynb)

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

# **Satellite Image Classification using Scikit-Learn**
**Author: Krishnagopal Halder**<br>

In this project, I harness the power of Scikit-Learn's Random Forest algorithm to develop a satellite image classification system. My goal is to automatically classify diverse land cover types within satellite images, including urban areas, water bodies, vegetation, and grass. I begin by collecting and preprocessing a labeled dataset, ensuring consistency in quality and format. To enhance model performance and interpretability, I employ Min-Max scaling to normalize feature values. Random Forest model is trained on these scaled features and evaluated using various metrics, with a particular focus on understanding feature importance, which helps us gain insights into the driving factors behind classification decisions. This project not only delivers an accurate image classification solution but also provides valuable insights into the significant features contributing to land cover classification, making it applicable in diverse fields such as environmental monitoring, urban planning, and land use land cover classification.


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

## **Read the Imagery**

```python
dataset = rasterio.open(r"D:\Research Works\Dataset\Raster\Landsat_8_Image_Kolkata.tif")
```

```python
# Visualize the 'RGB' and 'SFCC' image

# Create a function to Normalize the bands
def normalize(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min) / (band_max - band_min))

# Apply the normalize function over the bands
band_5 = normalize(dataset.read(5))
band_4 = normalize(dataset.read(4))
band_3 = normalize(dataset.read(3))
band_2 = normalize(dataset.read(2))

# Create the 'RGB' and 'SFCC' Image
rgb = np.dstack((band_4, band_3, band_2))
sfcc = np.dstack((band_5, band_4, band_3))

# Display the images
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 15))
ax1.imshow(rgb)
ax1.set_title("RGB Image of ROI")

ax2.imshow(sfcc)
ax2.set_title("SFCC Image of ROI")
plt.show()
```

```python
# Creating a empty pandas dataframe to store the pixel values
dataset_bands = pd.DataFrame()
```

```python
# Joining the pixel values of different bands into the dataframe
for i in dataset.indexes:
    temp = dataset.read(i)
    temp = pd.DataFrame(data=np.array(temp).flatten(), columns=[i])
    dataset_bands = temp.join(dataset_bands)
```

```python
dataset_bands
```

```python
# Rename the columns
new_column_names = {1:"Coastal", 2:"Blue", 3:"Green", 4:"Red", 5:"NIR", 6:"SWIR1",
                    7:"SWIR2", 8:"NDVI", 9:"MNDWI", 10:"NDBI", 11:"SAVI", 12:"Label"}
dataset_bands.rename(columns=new_column_names, inplace=True)
```

```python
dataset_bands
```

```python
# Changing the order of columns
dataset_bands = dataset_bands[dataset_bands.columns[::-1]]
dataset_bands
```

## **Data Preprocessing**


### **Apply MinMax Scaler**

```python
from sklearn.preprocessing import MinMaxScaler
```

```python
# Create an object of the minmax scaler
scaler = MinMaxScaler()

# Fit the data
scaler.fit(dataset_bands.drop("Label", axis=1))

# Transform the data
dataset_bands_scaled = scaler.transform(dataset_bands.drop("Label", axis=1))
```

```python
# Convert the scaled array into pandas dataframe
dataset_bands_scaled = pd.DataFrame(dataset_bands_scaled, columns=dataset_bands.iloc[:, :-1].columns)
dataset_bands_scaled
```

```python
# Describe the scaled data
dataset_bands_scaled.describe()
```

## **Exploratory Data Analysis (EDA)**

```python
# Plot the Histogram of all the bands
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 20))

# Flatten the axes array to make it easier to access each subplot
axes = axes.flatten()

# Loop through each column and plot its histogram in a subplot
for i, column in enumerate(dataset_bands_scaled.columns):
    ax = axes[i]
    sns.histplot(dataset_bands_scaled[[column]], ax=ax, bins=50)
    ax.set_title(column, fontsize=14)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequecy")
    
# Adjust spacing between subplots
plt.tight_layout()
plt.show()
```

```python
# Plot a heatmap to represent the correlation between bands
plt.figure(figsize=(8, 6))

# Create a mask to hide the upper triangle of the heatmap
mask = np.triu(np.ones_like(dataset_bands_scaled.corr()))
sns.heatmap(dataset_bands_scaled.corr(), cmap="RdYlGn", mask=mask, 
            linewidth=1, annot=True, fmt=".1f")
plt.title("Correlation Matrix")
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()
```

## **Prepare the Training Data**

```python
# Create the labeled data
dataset_bands_scaled["Label"] = dataset_bands["Label"]
dataset_bands_scaled
```

```python
# Extract the label data with non null values
labeled_data = dataset_bands_scaled.dropna()
labeled_data.head()
```

```python
# Change the datatype of the 'Label' column
labeled_data["Label"] = labeled_data["Label"].astype(int)
labeled_data.head()
```

```python
# Plot the training data
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(sfcc, alpha=0.5)
ax.imshow(np.array(dataset_bands_scaled["Label"]).reshape(dataset.shape), cmap="jet")
plt.title("Training Samples")
plt.show()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
x_train, x_test, y_train, y_test = train_test_split(labeled_data.drop("Label", axis=1),
                                                    labeled_data["Label"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

```python
x_train
```

## **Train a Classifier**

```python
from sklearn.ensemble import RandomForestClassifier
```

```python
# Create an object of the classifir
classifier = RandomForestClassifier(n_estimators=75,
                                    criterion="gini",
                                    max_depth=8,
                                    min_samples_split=10,
                                    random_state=0)

# Fit the training data
classifier.fit(x_train, y_train)
```

## **Feature Importance**

```python
# Extract the feature importance
feature_importance = dict(zip(x_train.columns, classifier.feature_importances_.round(3)))
# Convert the dictionary into a pandas series
feature_importance = pd.Series(feature_importance)
```

```python
# Sort the feature importance in descending order
feature_importance.sort_values(ascending=False, inplace=True)
```

```python
feature_importance
```

```python
# Plot the feature importance
plt.figure(figsize=(8, 6))

# Reverse the color palette 
color_palette = sns.color_palette("Reds", len(feature_importance))
reversed_palette = color_palette[::-1]

sns.barplot(x=feature_importance, y=feature_importance.index, palette=reversed_palette)
plt.title("Feature Importance")
plt.xlabel("Feature Importance")
plt.show()
```

## **Model Validation**

```python
# Predict the test data using RF Classifier 
y_pred = classifier.predict(x_test)
```

```python
from sklearn.metrics import classification_report, confusion_matrix
```

```python
# Print the model validation metrics
print(classification_report(y_test, y_pred))
```

```python
# Plot the confusion matrix
labels = ["Built-Up", "Water", "Vegetation", "Grass"]
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), cmap="YlGn", annot=True, fmt="",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
```

## **Classify the Imagery**

```python
# Classify the scaled imagery
classified = classifier.predict(dataset_bands_scaled.drop("Label", axis=1))
```

```python
# Check image dimensions
dataset.shape
```

```python
# Reshape the array into image dimensions
classified_array = classified.reshape(dataset.shape)
```

```python
# Plot the SFCC and classified array
import matplotlib.colors as mcolors

# Create a list of color
lulc_color = ["#EB5353", "#1450A3", "#285430", "#F9D923"]
cmap_custom = mcolors.ListedColormap(lulc_color)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 15))

ax1.imshow(sfcc)
ax1.set_title("Standard False Color Composite")

ax2.imshow(classified_array, cmap=cmap_custom)
ax2.set_title("Random Forest Classified Image")

# Show the figure
plt.show()
```

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/01_ANN/05_Graduate_Admission_Prediction_using_ANN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="XCjRLaOR68CT" -->
# **Graduate Admission Prediction using ANN**
<!-- #endregion -->

<!-- #region id="PmOeIf10D-Vr" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PI_1_6_3EemW" outputId="5edec3a2-47c5-4801-f735-887a7e21c4a4"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="9v9DoBH0ECBE"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="F4-y0zuyER4E" -->
## **Read the Data from Kaggle**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DSsQvk57EZoF" outputId="4ef531a4-f86a-477f-bca3-b3533f13950b"
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
```

```python colab={"base_uri": "https://localhost:8080/"} id="fLQUdsG1EpBg" outputId="e0011507-ca95-46f4-c383-eb9ed7d3571a"
# Download the data from kaggle
!kaggle datasets download -d mohansacharya/graduate-admissions
```

```python id="nWHgjEhxEvVP"
# Extract the Zipfile
import zipfile
zipref = zipfile.ZipFile("/content/graduate-admissions.zip")
zipref.extractall("/content")
zipref.close()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 223} id="pmiO1gM8FI0j" outputId="af8112f0-0067-4a7e-9804-2444e8f6636d"
# Read the data in a pandas dataframe
df = pd.read_csv("/content/Admission_Predict_Ver1.1.csv")
print(df.shape)
df.head()
```

<!-- #region id="A-dHcaNBFw7g" -->
## **Data Preprocessing**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nYPq31F8FbZG" outputId="23c5a23b-adff-41a7-c72b-c67738d6451c"
# Check the information of the columns
df.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="60FBZB6IFhwP" outputId="06acc1fa-e249-429b-dc98-bea6f6a03245"
# Check for duplicated rows
df.duplicated().sum()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="4EYrPBiaFsPW" outputId="5e88ae9d-42ce-473c-e58a-b34b8d748352"
# Drop the 'Serial No.' column
df.drop("Serial No.", axis=1, inplace=True)
df.head()
```

<!-- #region id="6OZPUVUWGEjp" -->
## **Train Test Split**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2UO7LK1UGM3D" outputId="bfc2d699-5125-4a46-ad5f-d4319366e5b8"
X_train, X_test, y_train, y_test = train_test_split(df.drop("Chance of Admit ", axis=1),
                                                    df["Chance of Admit "],
                                                    test_size=0.2,
                                                    random_state=0)

X_train.shape, X_test.shape
```

<!-- #region id="Wau-pEp9G0YY" -->
## **Feature Scaling**
<!-- #endregion -->

```python id="_MGTn1nsG4N5"
# Apply MinMaxScaler to all the columns

# Instantiate a MinMaxScaler object
minmax_scaler = MinMaxScaler()

# Fit the training data
minmax_scaler.fit(X_train)

# Transform the training and testing data
X_train_scaled = minmax_scaler.transform(X_train)
X_test_scaled = minmax_scaler.transform(X_test)
```

```python id="psHZUAOpHcA1"
# Convert the scaled arrays into pandas dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="mFopptKGHtyn" outputId="55c1a69f-a11f-48e5-a1d5-221373aabb74"
X_train_scaled.head()
```

<!-- #region id="CQPZK8-pHyD2" -->
## **Build an Artifical Neural Network Architecture**
<!-- #endregion -->

```python id="MRRZXN0AIJSB"
# Instantiate a Sequential model
model = Sequential()

# Add a Dense layer with 7 nodes into the Sequential Model
model.add(Dense(7, activation="relu", input_dim=7))
model.add(Dense(7, activation="relu"))
model.add(Dense(1, activation="linear"))
```

```python colab={"base_uri": "https://localhost:8080/"} id="eWnjFEPOIdF8" outputId="12e07678-c4d7-41d0-cce9-c77083613ec3"
# Print the model summary
model.summary()
```

```python id="zFVUbNS_ImD0"
# Compile the model
model.compile(loss="mean_squared_error", optimizer="Adam")
```

```python colab={"base_uri": "https://localhost:8080/"} id="IkOfSSmCI3cf" outputId="8a333ac0-e242-4dab-9b4b-fa5a683083e6"
# Fit the training data
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)
```

<!-- #region id="5ChbjIdaJG_v" -->
## **Accuracy Assessment**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="G96p2s68JTnq" outputId="da9cf1aa-e56b-4db5-f672-16a9e4f42d2f"
# Predict the test data
y_pred = model.predict(X_test_scaled)
# y_pred
```

```python colab={"base_uri": "https://localhost:8080/"} id="IcaO3dnEJZy6" outputId="68acc7e4-4347-4434-fc20-515f223dabbb"
# Check the accurcay of the model
print("R2 Score:", r2_score(y_test, y_pred))
```

<!-- #region id="gBRwOiIfKQLg" -->
## **Plot the Training and Validation Loss**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Z3BC5pcmKjqT" outputId="50029304-2941-4c6f-b386-e06c2f120ade"
# Print the key names of the history dictionary
history.history.keys()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="tQR2-R2CKdYK" outputId="79e22e0c-7037-4557-ba10-717e0a36847e"
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

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
    language: python
    name: python3
---

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/01_ANN/03_Customer_Churn_Prediction_using_ANN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="QKP2jG3pxW5p" -->
# **Customer Churn Prediction using ANN**

<img src="https://www.cleartouch.in/wp-content/uploads/2022/11/Customer-Churn.png">
<!-- #endregion -->

<!-- #region id="Z6ddHxd1xW5r" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ruF49494xq2S" outputId="eab9d0d9-235b-43c1-e35d-e9f450571ee6"
from google.colab import drive
drive.mount("/content/drive")
```

```python colab={"base_uri": "https://localhost:8080/"} id="XzuuFrixxW5r" outputId="15604840-fda2-4955-d828-ed332106c330"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
print(tf.__version__)
```

<!-- #region id="A9EY7R4zxW5r" -->
## **Read the Data from Kaggle**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Ts3ieKnByn1L" outputId="8fa4b258-9925-4cf0-95e0-fe2e5009dcae"
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
```

```python id="3NorY02KztVq" colab={"base_uri": "https://localhost:8080/"} outputId="8b55be0c-70d4-46e0-c025-ce508b6a32e9"
# Download the data from kaggle
!kaggle datasets download -d rjmanoj/credit-card-customer-churn-prediction
```

```python id="E6hhwo7jz7Pt"
# Extract the data from Zipfile
import zipfile
zipref = zipfile.ZipFile("/content/credit-card-customer-churn-prediction.zip")
zipref.extractall("/content")
zipref.close()
```

```python id="7ptGHXcWxW5r" outputId="de765b7b-9ad4-4bff-9fce-1c531670ebc2" colab={"base_uri": "https://localhost:8080/", "height": 243}
# Read the data in a pandas dataframe
df = pd.read_csv("/content/Churn_Modelling.csv")
print(df.shape)
df.head()
```

<!-- #region id="JIo5bdgNxW5r" -->
## **Data Preprocessing**
<!-- #endregion -->

```python id="_Vwg3rcaxW5s" outputId="1e7e7286-5410-4a97-e776-de99271be724" colab={"base_uri": "https://localhost:8080/"}
# Check for the missing values
df.isnull().sum()
```

```python id="j60XxGiOxW5s" outputId="929e4996-d471-4732-da41-c0f5bb699e45" colab={"base_uri": "https://localhost:8080/"}
# Check the information of all the columns
df.info()
```

```python id="VFGwdg8fxW5s" outputId="dab55928-a14f-42ca-bc71-f14825fe8012" colab={"base_uri": "https://localhost:8080/", "height": 206}
# Drop the irrelevant columns (e.g., columns with 'object' datatype)
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)
df.head()
```

```python id="i2ZAFvytxW5s" outputId="ff04f31e-266d-4d01-9f73-7b3ee1713261" colab={"base_uri": "https://localhost:8080/"}
# Check for the duplicated rows
df.duplicated().sum()
```

<!-- #region id="YJBjrHOS1mNi" -->
## **Exploratory Data Analysis**
<!-- #endregion -->

```python id="RzOpgSrpxW5s" outputId="491125cb-783a-46cd-8fff-a6aba3a99de4" colab={"base_uri": "https://localhost:8080/"}
# Check the number of people who Exited
df["Exited"].value_counts()
```

```python id="HPpLJfJgxW5s" outputId="ad2431aa-f2d1-4ec8-8152-8620dc366e9a" colab={"base_uri": "https://localhost:8080/", "height": 449}
# Plot the categorical variables
sns.countplot(x=df["Gender"], hue=df["Gender"])
plt.show()
```

```python id="ysWpi3rwxW5t" outputId="84503550-7d76-41ce-c2a2-11aaeac7d00e" colab={"base_uri": "https://localhost:8080/", "height": 449}
sns.countplot(x=df["Geography"], hue=df["Geography"])
plt.show()
```

<!-- #region id="rqsgzJGC17OM" -->
## **One Hot Encoding**
<!-- #endregion -->

```python id="Ys9fqlNNxW5t" outputId="655fb330-63d0-422a-8467-ba60da6ab08a" colab={"base_uri": "https://localhost:8080/", "height": 226}
# Apply One Hot Encoding on Categorical columns
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True, dtype=int)
df.head()
```

<!-- #region id="reFiLR77xW5t" -->
## **Train Test Split**
<!-- #endregion -->

```python id="LYB5JC_2xW5t" outputId="5afcd739-c7cb-4d82-c8eb-6cd822ba56e9" colab={"base_uri": "https://localhost:8080/"}
X_train, X_test, y_train, y_test = train_test_split(df.drop("Exited", axis=1),
                                                    df["Exited"],
                                                    test_size=0.2,
                                                    random_state=0)
X_train.shape, X_test.shape
```

<!-- #region id="uzCZhZX5xW5t" -->
## **Feature Scaling**
<!-- #endregion -->

```python id="1u_5il5mxW5t"
# Create an object of the StandardScaler class
scaler = StandardScaler()

# Fit the training data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python id="Av0iZ_57xW5u"
# Convert the scaled array into a pandas dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 243} id="UrwtceQH2sev" outputId="ccc2a84a-2f2b-4bc3-a4c2-402abab80b30"
print(X_train_scaled.shape)
X_train_scaled.head()
```

<!-- #region id="9cSMS4mHxW5u" -->
## **Build an Artifical Neural Network Architecture**
<!-- #endregion -->

```python id="mTO4h2mZxW5u"
# Instantiate a Sequential model
model = Sequential()

# Add two Dense layer with 11 nodes into the Sequential Model
model.add(Dense(11, activation="relu", input_dim=11))
model.add(Dense(11, activation="relu"))

# Add another layer for the output with a single node
model.add(Dense(1, activation="sigmoid"))
```

```python id="d0hUj9J7xW5u" outputId="eed36020-c1a3-4f4c-f28b-90bdf73c7079" colab={"base_uri": "https://localhost:8080/"}
# Print the model summary
model.summary()
```

```python id="7n6AtBN6xW5u"
# Compile the model
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
```

```python id="LqTqTr_kxW5u" outputId="7ced0e38-f51a-4d2c-e992-72a1fb6ba8f1" colab={"base_uri": "https://localhost:8080/"}
# Fit the training data
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)
```

```python id="_ZtkE8hExW5u" outputId="025e504b-aaac-4101-8f71-9532deb7e8a4" colab={"base_uri": "https://localhost:8080/"}
# Check all 132 weights of the first layer
model.layers[0].get_weights()[0]
```

```python id="SjeOCBu4xW5u" outputId="e19616ee-bed9-419e-c806-0dfa76064690" colab={"base_uri": "https://localhost:8080/"}
# Check all the weights of the second layer
model.layers[1].get_weights()
```

```python id="Ept_j2dixW5v" outputId="ff692cad-f91f-43b4-d304-14d0b5ff2a7c" colab={"base_uri": "https://localhost:8080/"}
# Check all the weights of the last layer
model.layers[2].get_weights()
```

<!-- #region id="oV_7VfvVxW5v" -->
## **Predict the Test Data**
<!-- #endregion -->

```python id="nMBTwbDrxW5v" outputId="ced56437-0974-486b-9a79-639f2d722262" colab={"base_uri": "https://localhost:8080/"}
y_pred = model.predict(X_test_scaled)
y_pred
```

```python id="-a9bkGsOxW5v" outputId="6457bec6-dfaa-4f1b-c1fe-abbe03de05d5" colab={"base_uri": "https://localhost:8080/"}
# Convert the predicted probability into binary classes
# Assume a threshold value
threshold = 0.5

y_pred = np.where(y_pred > threshold, 1, 0)
y_pred
```

<!-- #region id="FGinDmm0xW5v" -->
## **Accuracy Assessment**
<!-- #endregion -->

```python id="4kZaBhNxxW5v" outputId="1a66f230-3c80-4604-d0c7-86909af852b5" colab={"base_uri": "https://localhost:8080/"}
print("Accuracy:", accuracy_score(y_test, y_pred))
```

<!-- #region id="QaLEbC1sxW5v" -->
## **Plot the Training and Validation Loss**
<!-- #endregion -->

```python id="s7Q3eTs-xW51" outputId="fee43287-dad7-423a-e6d5-6ef903b2a510" colab={"base_uri": "https://localhost:8080/"}
# Print the key names of the history dictionary
history.history.keys()
```

```python id="LkC2pPwaxW52" outputId="e6e392e0-5211-4b86-b7ba-6135f2fa19d2" colab={"base_uri": "https://localhost:8080/", "height": 449}
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

```python id="paCJuouixW52" outputId="897031e1-92e2-490b-9431-c43bc580bccf" colab={"base_uri": "https://localhost:8080/", "height": 449}
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

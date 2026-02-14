[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/End-to-End-Deep-Learning/01_ANN/04_Handwritten_Digit_Classification.ipynb)

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

<!-- #region id="a-x0K-Z-fA6q" -->
# **Handwritten Digit Classification**
<!-- #endregion -->

<!-- #region id="094sdJYaPb5V" -->
## **Import Required Libraries**
<!-- #endregion -->

```python id="FuYtaLqeMgwr" executionInfo={"status": "ok", "timestamp": 1701200358984, "user_tz": -330, "elapsed": 414, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="JnZJsvtAPfPU" -->
## **Load MNIST Handwritten Digit Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="iAItE_k_PkWc" executionInfo={"status": "ok", "timestamp": 1701200359995, "user_tz": -330, "elapsed": 559, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="3873f7bf-a5c6-4148-b8ce-69e43bbdf0c9"
# Store the MNIST data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="nFf8MU9RQEUQ" executionInfo={"status": "ok", "timestamp": 1701200359995, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="c37a82d5-dce5-4a5d-b84f-50009cd6936f"
# Plot a digit
plt.imshow(X_train[1, :, :])
plt.show()
```

<!-- #region id="lVl0O7P1RnlC" -->
## **Scale the Pixel Values**
<!-- #endregion -->

```python id="bbl55WOTRthr" executionInfo={"status": "ok", "timestamp": 1701200359996, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}}
# Normalize the values
X_train = X_train / 255
X_test = X_test / 255
```

<!-- #region id="9bfUexrqSNSP" -->
## **Build an ANN Model**
<!-- #endregion -->

```python id="O607FUmBSm7j" executionInfo={"status": "ok", "timestamp": 1701200359996, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}}
# Initialize a model
model = Sequential()

model.add(Flatten(input_shape=(28, 28)))    # Input Layer
model.add(Dense(128, activation="relu"))    # First Hidden Layer
model.add(Dense(64, activation="relu"))     # Second Hidden Layer
model.add(Dense(32, activation="relu"))     # Third Hidden Layer
model.add(Dense(10, activation="softmax"))  # Output Layer
```

```python colab={"base_uri": "https://localhost:8080/"} id="JMOOpujdTgX9" executionInfo={"status": "ok", "timestamp": 1701200359996, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="8d5719a2-c878-44ee-8ec5-71774972da06"
# Summarize the model
model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="w1TkQ7BZUUoX" executionInfo={"status": "ok", "timestamp": 1701200557398, "user_tz": -330, "elapsed": 143583, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="ab621775-458e-4398-faf6-07d522d7cffe"
# Compile the model
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Fit the training data
history = model.fit(X_train, y_train, epochs=25, validation_split=0.2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="12aPbcZGVHV6" executionInfo={"status": "ok", "timestamp": 1701200559908, "user_tz": -330, "elapsed": 1119, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="77cddb43-bc9a-4d28-f342-c00198c4c691"
# Predict the test data
y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
y_pred
```

<!-- #region id="8TYRF7WOVtig" -->
## **Accuracy Score**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DH_1tKxYV4eJ" executionInfo={"status": "ok", "timestamp": 1701200559909, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="61de407b-e4e0-41c0-d11f-cfc22c284d90"
# Calculate the accuracy of model
accuracy = accuracy_score(y_test, y_pred)
accuracy
```

```python colab={"base_uri": "https://localhost:8080/", "height": 489} id="MHgKYnbsaEY9" executionInfo={"status": "ok", "timestamp": 1701200559909, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="2d0df916-676e-4c45-d357-4b3e496ca85b"
# Plot the 'Loss' with respect to 'Epochs'
sns.lineplot(history.history["loss"])
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 452} id="mICitVA_aRvP" executionInfo={"status": "ok", "timestamp": 1701200560895, "user_tz": -330, "elapsed": 992, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="804f0b21-68b7-4132-eaa2-f9f8d3e47ded"
# Plot the training loss with respect to validation loss
sns.lineplot(history.history["loss"], label="Training Loss")
sns.lineplot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 452} id="upknp43pcvrm" executionInfo={"status": "ok", "timestamp": 1701200560895, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="1937772b-cb5b-46c3-85e7-7c205423b5dc"
# Plot the training loss with respect to validation loss
sns.lineplot(history.history["accuracy"], label="Training Accuracy")
sns.lineplot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.show()
```

<!-- #region id="PUdoGiFNdnr-" -->
## **Model Prediction**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 465} id="TzRXFeA6dtFX" executionInfo={"status": "ok", "timestamp": 1701200970829, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="127e44d9-039f-45d1-db0d-9239995b3c89"
# Plot a random image from the test data
plt.imshow(X_test[100])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Lelxw5lkd0nI" executionInfo={"status": "ok", "timestamp": 1701201042127, "user_tz": -330, "elapsed": 2, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}} outputId="0bbd32c6-0963-44ce-a109-5bd54adc84db"
# Predict the image with the model
np.argmax(model.predict(X_test[100].reshape(1, 28, 28)), axis=1)
```

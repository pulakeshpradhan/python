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

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/01_ANN/00_Introduction_to_Perceptron.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="071e647b-a4f9-4718-8730-a8eac82b7dd0" -->
# **Introduction to Perceptron**
<!-- #endregion -->

<!-- #region id="e4f2a4f0-92b4-4c13-a4af-93c008270200" -->
A perceptron is one of the simplest and fundamental building blocks in deep learning and artificial neural networks. It was developed by Frank Rosenblatt in the late 1950s and is a type of artificial neuron or node that can be used for binary classification tasks. While perceptrons are limited in their capabilities compared to more complex neural network architectures, they serve as a foundational concept for understanding how neural networks work.
<!-- #endregion -->

<!-- #region id="1812e86e-4280-4096-9a07-7b21f17c05d5" -->
<center><img src="https://miro.medium.com/v2/resize:fit:1400/1*gGmqkjA0VJCe5EhJnoQDNg.png" width="50%"></center>
<!-- #endregion -->

<!-- #region id="e3ea96d0-39b8-45bf-8a64-a55ce7a868e0" -->
Here's an introduction to perceptrons in deep learning:

1. **Basic Structure**: A perceptron takes multiple binary inputs (0 or 1) and produces a single binary output (0 or 1). Each input is associated with a weight, and there is also an additional parameter called the bias. Mathematically, the output of a perceptron is calculated as the weighted sum of inputs plus the bias, followed by applying a step function (often the Heaviside step function or a similar activation function) to the sum.

$$y = \text{Activation Function}\left(\sum_{i=1}^{n} \text{weight}_i \cdot \text{input}_i + \text{bias}\right)$$


2. **Weights and Bias**: The weights in a perceptron represent the strength of the connection between the inputs and the output. A larger weight means that the corresponding input has a stronger influence on the output. The bias acts as an offset, allowing the perceptron to produce different outputs even when all inputs are zero.

3. **Activation Function**: The activation function determines whether the perceptron should fire (output 1) or not (output 0) based on the weighted sum of inputs plus the bias. The choice of activation function is crucial, as it introduces non-linearity into the model. Common activation functions include the step function, sigmoid, ReLU (Rectified Linear Unit), and others.
<!-- #endregion -->

<!-- #region id="4c1fab6a-cd13-457d-a943-65d86fbed1df" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1brKm_Yxvcgs" outputId="04f02813-7c06-469f-c7e7-74acff863910"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="75becf96-2a01-4b66-95f5-12a40768048c"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="8891e75c-77d3-4ffa-ab6d-32226090f4a2" -->
## **Read the Data**
<!-- #endregion -->

```python id="c77b28f6-6c15-4d65-9a7b-7874dcebd69a" outputId="c8394dff-0fb9-45c6-be3e-0d025c021186" colab={"base_uri": "https://localhost:8080/", "height": 423}
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/Placement.csv")
df
```

```python id="0bc3120c-d519-474e-abab-320b9860ab35" outputId="43c2e10a-ac1b-48cc-c0f1-d7231863b0b0" colab={"base_uri": "https://localhost:8080/"}
# Check if there is any null values
df.info()
```

<!-- #region id="6bbee747-a36b-4d5f-8ab0-6922b22fe664" -->
## **Train Test Split**
<!-- #endregion -->

```python id="073928d4-26cb-445a-80e1-ceedb0e9fc0f" outputId="ab3a038e-fd8c-4707-9a17-9db0136bced8" colab={"base_uri": "https://localhost:8080/"}
X_train, X_test, y_train, y_test = train_test_split(df.drop("placed", axis=1),
                                                    df["placed"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

<!-- #region id="e6171f13-7408-4145-9a90-f86401950ed3" -->
## **Data Visualization**
<!-- #endregion -->

```python id="89303e37-1bda-472f-8be1-31b8fb87cec0" outputId="88022cf8-ca64-458f-fe23-6ad504a172ab" colab={"base_uri": "https://localhost:8080/", "height": 472}
# Display a scatterplot between CGPA and IQ
sns.scatterplot(data=df, x="cgpa", y="resume_score", hue=df["placed"])
plt.title("Scatterplot between CGPA and IQ")
plt.show()
```

<!-- #region id="d198f3cd-6527-416f-8abd-66e5c407ab86" -->
## **Train a Perceptron for Classification**
<!-- #endregion -->

```python id="6237277f-3e1d-40c4-9d46-287b17772ee0" outputId="60a8ac3a-4d7a-4a6e-a9b9-cd1cd426aa9b" colab={"base_uri": "https://localhost:8080/", "height": 74}
# Create an object of the Perceptron class
perceptron = Perceptron(random_state=0)

# Fit the training data
perceptron.fit(X_train, y_train)
```

```python id="663ca708-746b-4b2a-98e9-01df30942769" outputId="31da0612-85ad-46bf-93bf-73689f2ebe69" colab={"base_uri": "https://localhost:8080/"}
# Check all the coefficients (weights)
perceptron.coef_
```

```python id="920f95ff-4f1f-4f93-a301-496fd0cfe91c" outputId="846d0cdb-c07e-40ef-d1a9-03bdc7965a6e" colab={"base_uri": "https://localhost:8080/"}
# Check the intercept (bias)
perceptron.intercept_
```

```python id="d05fce88-b34d-45d5-b9ff-3a588dc0ad9f" outputId="ede65587-e648-49ca-dd1e-a4c3f56d7562" colab={"base_uri": "https://localhost:8080/"}
# Predict the test data
y_pred = perceptron.predict(X_test)
y_pred
```

<!-- #region id="df5c7602-a33b-497d-8f94-2b3f06850a7b" -->
## **Accuracy Assessment**
<!-- #endregion -->

```python id="c9905b33-13fa-4e01-a386-68780db67622" outputId="180b85d1-b5c3-4da7-a7e7-2777bc4d6aa0" colab={"base_uri": "https://localhost:8080/"}
print("Accuracy of Perceptron Model:", accuracy_score(y_test, y_pred).round(2))
```

<!-- #region id="a07e65f4-7d3a-440d-8335-1c3a0bef6caf" -->
## **Display the Decision Boundary**
<!-- #endregion -->

```python id="f14f6a9d-5fe1-4765-b354-1f71352ab7f0" outputId="38d46188-c5be-4ad4-9388-c8583c2c74cf" colab={"base_uri": "https://localhost:8080/", "height": 449}
plot_decision_regions(X_train.values, y_train.values, clf=perceptron, legend=2)
```

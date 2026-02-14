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
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/01_ANN/06_Backpropagation_on_Regression_Data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="c55377a4-5ba0-40c9-9caf-b581ebf5edb0" -->
# **Backpropagation on Regression Data**
Backpropagation, short for "backward propagation of errors," is a supervised learning algorithm used to train artificial neural networks, which are a fundamental component of deep learning. The purpose of backpropagation is to minimize the error between the predicted output of the neural network and the actual target output by adjusting the network's weights and biases.

Here's a step-by-step explanation of how backpropagation works:

1. **Forward Pass:**
   - The input data is passed through the neural network to generate a predicted output.
   - Each neuron in the network performs a weighted sum of its inputs, applies an activation function to the result, and passes the output to the next layer.

2. **Calculate Error:**
   - The predicted output is compared to the actual target output to calculate the error. Common loss functions, such as mean squared error or cross-entropy, are used for this purpose.

3. **Backward Pass (Backpropagation):**
   - The error is then propagated backward through the network to update the weights and biases.
   - The partial derivative of the error with respect to each weight and bias is computed using the chain rule of calculus. This indicates how much the error would increase or decrease with a small change in each weight and bias.
   - The weights and biases are adjusted in the opposite direction of the gradient to minimize the error.

4. **Update Weights and Biases:**
   - The learning rate is applied to control the size of the weight and bias updates.
   - The weights and biases are updated based on the calculated gradients, nudging them towards values that reduce the error.

5. **Iterative Process:**
   - Steps 1-4 are repeated iteratively for multiple epochs until the neural network converges to a set of weights and biases that minimize the error on the training data.

The backpropagation algorithm allows neural networks to learn from data by adjusting their parameters to improve performance. It's a crucial component of training deep learning models and has been instrumental in the success of various applications, including image and speech recognition, natural language processing, and many others.
<!-- #endregion -->

<!-- #region id="Xe4QoAfxNF8h" -->
<center><img src="https://miro.medium.com/v2/resize:fit:1280/1*VF9xl3cZr2_qyoLfDJajZw.gif" width="50%"></center>

<!-- #endregion -->

<!-- #region id="faf27561-d783-426a-8237-6ba70c70e910" -->
## **Import Required Libraries**
<!-- #endregion -->

```python id="0f697383-a9ed-453d-aa2c-42fb9ca34af8" colab={"base_uri": "https://localhost:8080/"} outputId="782015cc-269d-4715-b762-9d218870538d"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
print(tf.__version__)

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="13508119-6857-4a91-8dca-4f90ec6d136d" -->
## **Create a DataFrame**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 175} id="2ed4b0c6-ce7b-4043-bf62-9b1660c85f5e" outputId="3a404d18-dd43-4825-e0d9-062c63283eb4"
df = pd.DataFrame([[8, 8, 4], [7, 9, 5], [6, 10, 6], [5, 12, 17]], columns=["cgpa", "profile_score", "lpa"])
df
```

<!-- #region id="ci_ZPFnjrOnB" -->
## **Build the Algorithm**
<!-- #endregion -->

```python id="2db41aab-5201-4eae-ab84-fb8e279b6eed"
# Create a function to initialize the weights and biases
def initialize_parameters(layer_dims):

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):

        parameters["W" + str(l)] = np.ones((layer_dims[l-1], layer_dims[l])) * 0.1
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters
```

```python colab={"base_uri": "https://localhost:8080/"} id="7QtwfZySVC1P" outputId="b7a6d307-e740-4ca7-e3c4-24877cd866c7"
# Check the function
initialize_parameters([2, 2, 1])
```

```python id="U6Y3QX4xVISo"
# Write a function to linearly forward the data
def linear_forward(A_prev, W, b):

    Z = np.dot(W.T, A_prev) + b

    return Z
```

```python id="-6s9lhPmW9Yk"
# Write a function for Forward Propagation
def L_layer_forward(X, parameters):

    A = X
    L = len(parameters) // 2 # Number of layers in the neural network

    for l in range(1, L+1):
        A_prev = A
        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]
        # print("A"+str(l-1)+": ", A_prev)
        # print("W"+str(l)+": ", Wl)
        # print("b"+str(l)+": ", bl)
        # print("__"*20)

        A = linear_forward(A_prev, Wl, bl)
        # print("A" + str(l)+": ", A)
        # print("**"*20)

    return A, A_prev
```

```python id="G44qyZseaTkN"
# Extract the X and y value of a particular student
X = df[["cgpa", "profile_score"]].values[0].reshape(2, 1) # Shape(no of features, no. of training example)
y = df[["lpa"]].values[0][0]

# Initialize the parameters
parameters = initialize_parameters([2, 2, 1])

# Apply forward propagation on that particular student
y_hat, A1 = L_layer_forward(X, parameters)
```

```python colab={"base_uri": "https://localhost:8080/"} id="kLOHTObMj7Ks" outputId="b8d8161a-c2d6-49a0-bf2a-c1a8ec5ed746"
y_hat = y_hat[0][0]
print(y_hat)
print(A1)
print("Loss:", (y-y_hat)**2)
```

```python id="vm6j9tA7dSMT"
# Write a function to update the parameters
def update_parameters(parameters, y, y_hat, A1, X, lr=0.001):
    parameters["W2"][0][0] = parameters["W2"][0][0] + (lr * 2 * (y - y_hat) * A1[0][0])
    parameters["W2"][1][0] = parameters["W2"][1][0] + (lr * 2 * (y - y_hat) * A1[1][0])
    parameters["b2"][0][0] = parameters["b2"][0][0] + (lr * 2 * (y - y_hat))

    parameters["W1"][0][0] = parameters["W1"][0][0] + (lr * 2 * (y - y_hat) * parameters["W2"][0][0] * X[0][0])
    parameters["W1"][0][1] = parameters["W1"][0][1] + (lr * 2 * (y - y_hat) * parameters["W2"][0][0] * X[1][0])
    parameters["b1"][0][0] = parameters["b1"][0][0] + (lr * 2 * (y - y_hat) * parameters["W2"][0][0])

    parameters["W1"][1][0] = parameters["W1"][1][0] + (lr * 2 * (y - y_hat) * parameters["W2"][1][0] * X[0][0])
    parameters["W1"][1][1] = parameters["W1"][1][1] + (lr * 2 * (y - y_hat) * parameters["W2"][1][0] * X[1][0])
    parameters["b1"][1][0] = parameters["b1"][1][0] + (lr * 2 * (y - y_hat) * parameters["W2"][1][0])
```

```python colab={"base_uri": "https://localhost:8080/"} id="zlYntRM7jSEo" outputId="3d281964-11fc-48b6-a77f-9623e026cf4c"
# Apply the function to update the parameters value
update_parameters(parameters, y, y_hat, A1, X, lr=0.001)
parameters
```

```python colab={"base_uri": "https://localhost:8080/"} id="JXzmD9qUkUU3" outputId="49f11c6e-f813-4d26-f1b2-dc3012c22394"
# Epoch implementation
parameters = initialize_parameters([2, 2, 1])
epochs = 5

for i in range(epochs):

    Loss = []

    for j in range(df.shape[0]):
        X = df[["cgpa", "profile_score"]].values[j].reshape(2, 1)
        y = df[["lpa"]].values[j][0]

        # Parameter initialization
        y_hat, A1 = L_layer_forward(X, parameters)
        y_hat = y_hat[0][0]

        update_parameters(parameters, y, y_hat, A1, X)

        Loss.append((y - y_hat)**2)
    print(f"Epoch - {i+1}, Loss - {np.array(Loss).mean()}")

parameters
```

<!-- #region id="Akg-mdDzK1Wf" -->
## **Backpropagation in Keras**
<!-- #endregion -->

```python id="SOfw887DK84-"
# Build the same model architecture using Keras
model = Sequential()

model.add(Dense(2, activation="linear", input_dim=2))
model.add(Dense(1, activation="linear"))
```

```python colab={"base_uri": "https://localhost:8080/"} id="GIr5rBg3LnFN" outputId="c56b8697-80cc-446b-f398-0ea10fcdfd7f"
# Print the summary of the model
model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="TKr48G7YLkpW" outputId="199cca8a-75fb-4f83-eecd-c5d07d869d45"
# Cehck the value of random weights initialized by Keras
model.get_weights()
```

```python colab={"base_uri": "https://localhost:8080/"} id="IGG6ARBhL8We" outputId="47f1e72f-3024-4ca4-97e9-4c290be2c2cc"
# Set the initial weights to 0.1 and biases to 0
new_weights = [np.array([[0.1 , 0.1 ],
                         [0.1 , 0.1]], dtype=np.float32),
               np.array([0., 0.], dtype=np.float32),
               np.array([[ 0.1],
                         [0.1 ]], dtype=np.float32),
               np.array([0.], dtype=np.float32)]

model.set_weights(new_weights)
model.get_weights()
```

```python id="YwWVmyerNThv"
# Define the optimizer and compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss="mean_squared_error", optimizer=optimizer)
```

```python colab={"base_uri": "https://localhost:8080/"} id="wpFUnFk1R_rF" outputId="ee7a70d0-4bc7-4a50-d981-41f8be74ba82"
# Fit the training data
history = model.fit(df.iloc[:, 0:-1].values, df["lpa"].values, epochs=1000, verbose=1, batch_size=1)
```

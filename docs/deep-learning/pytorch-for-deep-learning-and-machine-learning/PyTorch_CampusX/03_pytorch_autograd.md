[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/pytorch-for-deep-learning-and-machine-learning/PyTorch_CampusX/03_pytorch_autograd.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: py310
    language: python
    name: python3
---

<!-- #region id="hxp2jwIqbI9S" -->
# **PyTorch Autograd**

Autograd is PyTorch's automatic differentiation engine. It keeps track of all operations performed on tensors that have `requires_grad=True`, creating a computation graph dynamically. This graph is used to compute gradients for optimization tasks like backpropagation.

### **How Does It Work?**
1. **Computation Graph:**
   - When you perform operations on tensors, autograd dynamically builds a directed acyclic graph (DAG) where nodes represent operations and edges represent the flow of data.
   - This graph allows autograd to trace how each tensor is derived from others.

2. **Backward Pass:**
   - When you call `.backward()` on a tensor, autograd traverses the graph in reverse order (hence "backpropagation"), computing gradients for all tensors with `requires_grad=True`.

3. **Gradient Storage:**
   - Gradients are stored in the `.grad` attribute of the corresponding tensor.
<!-- #endregion -->

<!-- #region id="6-eaB_SKkMZB" -->
## **Import Dependencies**
<!-- #endregion -->

```python executionInfo={"elapsed": 4689, "status": "ok", "timestamp": 1737239751080, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="R4cOlDb_kQwD"
import numpy as np
import matplotlib.pyplot as plt
import torch

import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="O86hSGhidee6" -->
## **Calculate Gradient Manually**

**Example-1:**<br>
We want to calculate the gradient of the function:

$$ y = x^2 $$

The derivative of \( y \) with respect to \( x \) is:

$$ \frac{\partial y}{\partial x} = 2x $$

Using Python, we can compute the gradient at a specific value of \( x \) with the following code:

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1737239751080, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="z3Gig1uUjqGv" outputId="64b47f94-1056-4261-dd5b-c7f5da02eec0"
# Function to calculate the gradient of y = x^2
def dy_dx(x):
    """
    Calculate the derivative of y = x^2 with respect to x.

    Parameters:
        x (float or int): The value of x at which the gradient is evaluated.

    Returns:
        float: The gradient (2 * x).
    """
    return 2 * x

# Example usage
x = 3
gradient = dy_dx(x)
print(f"The gradient of y = x^2 at x = {x} is {gradient}.")
```

<!-- #region id="pB2PBqDso1BR" -->
**Example-2:**<br>
We want to calculate the gradient of the function:
$$ y = x^2 $$
$$ z = \sin {y} $$

The derivative of \( z \) with respect to \( x \) is:

$$ \frac{\partial z}{\partial x} = \frac{∂z}{∂y} ⋅ \frac{∂y}{∂x} $$
$$ \frac{\partial z}{\partial x} = \cos{y} ⋅ 2x $$
$$ \frac{\partial z}{\partial x} = \cos{x^2} ⋅ 2x $$

Using Python, we can compute the gradient at a specific value of \( x \) with the following code:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1737239751080, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="SNefBd91pTt2" outputId="4453ec90-eace-482c-dd23-2b6de822db66"
# Function to calculate the gradient of z
import math

def dz_dx(x):
    """
    Calculate the derivative of z with respect to x.

    Parameters:
        y (float or int): The value of x at which the gradient is evaluated.

    Returns:
        float: The gradient.
    """
    return math.cos(x**2) * (2 * x)

# Example usage
x = 3
gradient = dz_dx(x)
print(f"The gradient of z at x = {x} is {gradient:.2f}.")
```

<!-- #region id="6XAw9PBatKqg" -->
## **Calculate Gradient using PyTorch**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1737239751080, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="TjupWz-4tQSn" outputId="b9333d21-a682-4de5-86c8-13c99993a5ab"
# Example-1
# Define a tensor with gradient tracking enabled
x = torch.tensor(3.0, requires_grad=True)

# Define the function y = x^2
y = x**2

# Print the values of x and y
print("x:", x)
print("y:", y)

# Perform backpropagation to compute the gradient
y.backward()

# Print the gradient of y with respect to x
print("Gradient (dy/dx):", x.grad)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1737239751080, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="0is1TmcVt_vP" outputId="4f6fa99c-341f-46c1-abff-958d4123cd48"
# Example-2
# Define a tensor with gradient tracking enabled
x = torch.tensor(3.0, requires_grad=True)

# Define the function y = x^2
y = x**2

# Define the function z = sin(y)
z = torch.sin(y)

# Print the values of x and y
print("x:", x)
print("y:", y)
print("z:", z)

# Perform backpropagation to compute the gradient
z.backward()

# Print the gradient of y with respect to x
print("Gradient (dy/dx):", x.grad)
```

<!-- #region id="hnzg-HZskgJ2" -->
## **Manual Gradient of Loss Calculation w.r.t Weight and Bias**

1. Linear Transformation:
$$ z = w \cdot x + b $$
2. Activation:
$$ y_{pred} = σ(z) = \frac{1}{1 + e^{-z}} $$
3. Loss Function (Binary Cross-Entropy Loss):
$$ L = -[y_{target} \cdot \ln(y_{pred}) + (1 - y_{target}) \cdot \ln( - y_{pred})] $$

<!-- #endregion -->

```python executionInfo={"elapsed": 218, "status": "ok", "timestamp": 1737240113448, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="F9rVMl9OkmOO"
# Inputs
x = torch.tensor(6.7) # Input feature
y = torch.tensor(0.0) # True Label (Binary)

w = torch.tensor(1.0) # Weight
b = torch.tensor(0.0) # Bias
```

```python executionInfo={"elapsed": 208, "status": "ok", "timestamp": 1737240941033, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="Id8yFk5TlHRh"
# Binary Cross-Entropy Loss for scalar
def binary_cross_entropy_loss(prediction, target):
    epsilon = 1e-8
    prediction = torch.clamp(prediction, epsilon, 1-epsilon)
    return -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 220, "status": "ok", "timestamp": 1737241262860, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="M317Df_uoTNo" outputId="da8f6ea1-5590-4de8-9bb5-143ff5721442"
# Forward pass
z = w * x + b # Weighted sum (Linear Transformation)
y_pred = torch.sigmoid(z) # Predicted Probability (Activation)

# Compute binary cross-entropy loss
loss = binary_cross_entropy_loss(y_pred, y)
print(loss)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1737241511871, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="7ZbT9UsOopoq" outputId="e68ee04d-ad31-41a3-ddeb-22cd45ee26e7"
# Derivatives:
# 1. dL/d(y_pred): Loss with respect to the prediction (y_pred)
dloss_dy_pred = (y_pred - y) / (y_pred * (1 - y_pred))

# 2. d(y_pred)/dz: Prediction (y_pred) with respect to z (sigmoid derivative)
dy_pred_dz = y_pred * (1 - y_pred)

# 3. dz/dw and dz/db: z with respect to w and b
dz_dw = x # dz/dw = x
dz_db = 1 # dz/db = 1 (bias contributes directly to z)

dL_dw = dloss_dy_pred * dy_pred_dz * dz_dw
dL_db = dloss_dy_pred * dy_pred_dz * dz_db
print(f"Manual Gradient of loss w.r.t weight (dw): {dL_dw:.4f}")
print(f"Manual Gradient of loss w.r.t bias (db): {dL_db:.4f}")
```

<!-- #region id="a2fQoQ2Booqy" -->
## **Automatic Gradient of Loss Calculation w.r.t Weight and Bias using Autograd**
<!-- #endregion -->

```python executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1737241624557, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="kqNYJc2PqrSw"
# Inputs
x = torch.tensor(6.7) # Input feature
y = torch.tensor(0.0) # True Label (Binary)

w = torch.tensor(1.0, requires_grad=True) # Weight
b = torch.tensor(0.0, requires_grad=True) # Bias
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 765, "status": "ok", "timestamp": 1737241655224, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="wbeE6290q6hS" outputId="ba92e823-d93f-40d4-fdb6-77430c2d0577"
# Forward pass
z = w * x + b # Weighted sum (Linear Transformation)
y_pred = torch.sigmoid(z) # Predicted Probability (Activation)

# Compute binary cross-entropy loss
loss = binary_cross_entropy_loss(y_pred, y)
print(loss)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1737241720252, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="2h5JdvgYrJNM" outputId="38c41093-495b-4ad2-f3e5-84be8666412f"
loss.backward()

print(w.grad)
print(b.grad)
```

<!-- #region id="v4o-9XY8srX1" -->
## **Calculate Gradients for Multiple Inputs**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 257, "status": "ok", "timestamp": 1737242652322, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="kjvJMULTs4Tn" outputId="16b29523-b469-4361-ca9f-268d686ad11c"
# Create a PyTorch tensor with multiple inputs
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x)
y = (x ** 2).mean()
print(y)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 250, "status": "ok", "timestamp": 1737242653175, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="PR2RB3AZtPAa" outputId="17851f38-c10b-4eaa-833a-fd8b9b0c9721"
y.backward()
x.grad
```

<!-- #region id="C5rHyDtRtlRr" -->
## **Clearing Gradients**
Gradients can be cleared using the `optimizer.zero_grad()` function when using optimizers. For manually tracking gradients, you can reset the gradients by assigning None to the .grad attribute of the tensor.
<!-- #endregion -->

```python executionInfo={"elapsed": 234, "status": "ok", "timestamp": 1737242656088, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="a-0-Z1mUt1c2"
x = torch.tensor(6.0, requires_grad=True)
y = (x ** 2)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 211, "status": "ok", "timestamp": 1737242657604, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="6DQp4tX7uRU5" outputId="48edb965-965c-44e4-fec0-f00a8bc92f63"
y.backward()
print(x.grad)
x.grad.zero_()
```

<!-- #region id="OeAwfEcTvC_X" -->
## **Disable Gradient Tracking**
n PyTorch, you can disable gradient tracking when gradients are not needed, such as during inference or evaluations, to improve computational efficiency. This is done using the `torch.no_grad() `context manager or by setting `requires_grad=False` for specific tensors.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 252, "status": "ok", "timestamp": 1737242857201, "user": {"displayName": "Krishnagopal Halder", "userId": "16954898871344510854"}, "user_tz": -60} id="xEsyVP6rvHXn" outputId="76743d51-0de3-4b3f-9160-6c264c6a8a61"
# Create tensors with gradient tracking enabled
x = torch.tensor(6.7, requires_grad=True)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Perform operations without gradient tracking
with torch.no_grad():
    z = w * x + b  # No gradients will be tracked for this operation
    y_pred = torch.sigmoid(z)

print(f"z: {z}")
print(f"y_pred: {y_pred}")

# Verify that gradients are not tracked
print(f"Requires Grad (z): {z.requires_grad}")  # Output: False
```

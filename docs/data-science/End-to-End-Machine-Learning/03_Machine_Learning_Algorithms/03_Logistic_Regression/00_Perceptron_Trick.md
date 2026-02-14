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

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/04_Machine_Learning_Algorithms/11_Perceptron_Trick.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="2f030272-ab15-4457-81be-ccf262d4276d" -->
# **Perceptron Trick**
<!-- #endregion -->

<!-- #region id="e7bbea89-64bc-4f55-816c-e289962e56fe" -->
A perceptron is one of the simplest and fundamental building blocks in deep learning and artificial neural networks. It was developed by Frank Rosenblatt in the late 1950s and is a type of artificial neuron or node that can be used for binary classification tasks. While perceptrons are limited in their capabilities compared to more complex neural network architectures, they serve as a foundational concept for understanding how neural networks work.
<!-- #endregion -->

<!-- #region id="451f1cbb-177f-4ee2-aa6a-0129288b3b79" -->
<center><img src="https://miro.medium.com/v2/resize:fit:1400/1*gGmqkjA0VJCe5EhJnoQDNg.png" width="50%"></center>
<!-- #endregion -->

<!-- #region id="570a1127-8286-43f3-8faa-47e6cf61ed61" -->
Here's an introduction to perceptrons in deep learning:

1. **Basic Structure**: A perceptron takes multiple binary inputs (0 or 1) and produces a single binary output (0 or 1). Each input is associated with a weight, and there is also an additional parameter called the bias. Mathematically, the output of a perceptron is calculated as the weighted sum of inputs plus the bias, followed by applying a step function (often the Heaviside step function or a similar activation function) to the sum.

$$y = \text{Activation Function}\left(\sum_{i=1}^{n} \text{weight}_i \cdot \text{input}_i + \text{bias}\right)$$

2. **Weights and Bias**: The weights in a perceptron represent the strength of the connection between the inputs and the output. A larger weight means that the corresponding input has a stronger influence on the output. The bias acts as an offset, allowing the perceptron to produce different outputs even when all inputs are zero.

3. **Activation Function**: The activation function determines whether the perceptron should fire (output 1) or not (output 0) based on the weighted sum of inputs plus the bias. The choice of activation function is crucial, as it introduces non-linearity into the model. Common activation functions include the step function, sigmoid, ReLU (Rectified Linear Unit), and others.
<!-- #endregion -->

<!-- #region id="6a21b674-fc16-45f8-838c-9253a0e5ae3a" -->
## **Import Required Libraries**
<!-- #endregion -->

```python id="2a6b1568-306f-4105-8ba0-c3a014073875"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="5c0e0687-e8b2-48b9-a003-d6f6a400b99a" -->
## **Make a Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="83591f0d-f6e7-4591-a4b6-abde048366c5" outputId="88d2482d-91d0-4664-faf7-bfc4a779c1f1"
# Make a sample classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2,
                           n_clusters_per_class=1, random_state=0, hypercube=False, class_sep=1.5)

# Plot the data
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
plt.show()
```

<!-- #region id="f358f452-6b76-412f-9d20-3a2c38b73296" -->
## **Build the Perceptron Algorithm**
<!-- #endregion -->

```python id="878142c1-a751-44a6-835d-bfa927ad07e7"
# Write a function to build the algorithm of a step function
def step(z):
    """
    This function returns 0 if value is less than or equals to 0 and returns 1
    if value is greater than 0.
    """
    return 0 if z <= 0 else 1
```

```python id="24c0abae-e61a-4353-a915-9e27009ec698"
# Write a function to build the algorithm of a perceptron
def perceptron(X, y, epochs):

    # Add an extra column for intercept term
    X = np.insert(X, 0, 1, axis=1)

    # Initialize the weights
    weights = np.ones(X.shape[1])

    # Initialize a learning rate
    lr = 0.01

    for i in range(epochs):
        # Select a random index
        n = np.random.randint(len(X))
        # Calculate the y-predicted
        y_hat = step(np.dot(X[n], weights))
        # Update the weights
        weights = weights + lr * (y[n]-y_hat) * X[n]

    return weights[0], weights[1:]
```

```python colab={"base_uri": "https://localhost:8080/"} id="561f2bbf-515d-4632-bbd3-a04bcd37f15c" outputId="24f78a20-79ea-4202-c3db-a1004f7ed3b3"
# Calculate the intercept and coefficients
intercept_, coef_ = perceptron(X, y, epochs=1000)

print("Intercept(w0):", intercept_)
print("Coefficients(w1, w2):", coef_)
```

<!-- #region id="a47deb8f-0f5a-4302-a19f-2c8f2e1f653a" -->
General Equation of a Line is:<br>
$$ \ Ax + By + C = 0 \ $$

We an also write it like:<br>
$$ \ y = mx + c \ $$ where $ \ m\ $ is the slope and $ \ c\ $ is the y-intercept.<br>
or,
$$ \ y = -\frac{A}{B}x - \frac{C}{B} \ $$
where $ -\frac{A}{B} \ $ is the slope and $ -\frac{C}{B} \ $ is the intercept.<br>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="85be8ff3-ddc9-4a31-92e8-066528790a9e" outputId="dfe73a20-2e6a-4a4b-bc10-d1df790d9ff2"
m = -(coef_[0]/coef_[1])
c = -(intercept_/coef_[1])
print("Slope(m):", m)
print("Y-Intercept(c):", c)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="c593620d-012c-4895-88d7-1553df7357ed" outputId="c297ffaf-2bd3-49dd-de43-e54060e07d1e"
# Plot the decision boundary
X_line = np.linspace(-1, 1, 50)
y_line = X_line * m + c

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
sns.lineplot(x=X_line, y=y_line, c="red", label="Decision Boundary")
plt.ylim((-2.5, 2.5))
plt.show()
```

<!-- #region id="f3ab6cbb-d3c1-4c99-9b22-aff819d6020a" -->
## **Apply the Logistic Regression**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 74} id="d6fa6dfd-ea12-436e-81e0-24febdb464f2" outputId="2faee3ff-6d15-4b3d-d277-eed660b92a42"
# Instantiate a Logistic Regression model
lr = LogisticRegression()

# Fit the data
lr.fit(X, y)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6c468496-495b-4ae7-b515-ef0b97d59f55" outputId="6c9403cb-dc80-448b-c960-d604446dd808"
# Print intercept and coefficients
print("Intercept of LR Model (w0):", lr.intercept_)
print("Coefficients of LR Model (w1, w2):", lr.coef_)
```

```python colab={"base_uri": "https://localhost:8080/"} id="a654fc03-16e6-4535-9c37-6dc404d481e0" outputId="cca29d24-60cc-4889-d2b4-9cb06a126a5f"
# Calculate the slope(m) and y-intercept(c) of the LR model
m_lr = -(lr.coef_[0][0] / lr.coef_[0][1])
c_lr = -(lr.intercept_[0] / lr.coef_[0][1])

print("Slope(m) of LR:", m_lr)
print("Y-Intercept(c) of LR:", c_lr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="5988bebf-ea8f-4efc-8645-2f035c846570" outputId="a8fc18e8-ec31-4659-9cca-c5e26af130e6"
# Plot the decision boundary
X_line = np.linspace(-1, 1, 50)
y_line_lr = X_line * m_lr + c_lr

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
sns.lineplot(x=X_line, y=y_line, c="red", label="Perceptron")
sns.lineplot(x=X_line, y=y_line_lr, c="green", label="Logistic Regression")
plt.ylim((-2.5, 2.5))
plt.show()
```

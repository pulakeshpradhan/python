[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/01_Gradient_Descent/04_Mini_Batch_Gradient_Descent.ipynb)

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

# **Mini-Batch Gradient Descent**
Mini-Batch Gradient Descent is a compromise between Stochastic Gradient Descent (SGD) and Batch Gradient Descent. In Mini-Batch Gradient Descent, the dataset is divided into small batches, and the model parameters are updated based on the average gradient of the loss function computed over each batch. This approach combines some of the advantages of both SGD and Batch Gradient Descent.


<center><img src="https://editor.analyticsvidhya.com/uploads/58182variations_comparison.png" style="width: 60%"></ceneter>


Here's how Mini-Batch Gradient Descent works:

1. **Initialization**: Start with an initial set of model parameters.

2. **Data Batching**: Divide the training dataset into small batches. The size of these batches is a hyperparameter known as the batch size.

3. **Data Shuffling**: Optionally, shuffle the batches to introduce some randomness and ensure that the order of batches doesn't bias the training.

4. **Iterative Updates**: For each training iteration (or epoch), process one mini-batch at a time. The model's parameters are updated based on the average gradient of the loss function computed over the data points in the mini-batch.

5. **Gradient Computation**: Compute the gradient of the loss function with respect to the model parameters by backpropagating errors through the network (for neural networks) or using analytical derivatives (for simpler models).

6. **Parameter Update**: Adjust the model parameters in the opposite direction of the average gradient. The learning rate controls the size of the step during each update.

7. **Repeat**: Steps 4-6 are repeated for each mini-batch in the dataset until convergence criteria are met.

Mini-Batch Gradient Descent has several advantages:

- **Efficiency**: It takes advantage of vectorized operations, making it more computationally efficient than pure Stochastic Gradient Descent, especially when implemented on hardware that is optimized for matrix operations (e.g., GPUs).

- **Regularization**: The mini-batch updates introduce a level of noise that can act as a form of regularization, potentially helping to prevent overfitting.

- **Parallelization**: Mini-Batch Gradient Descent allows for parallelization, as multiple mini-batches can be processed simultaneously.

- **Balanced Approach**: It strikes a balance between the high variance of SGD (processing one data point at a time) and the high computational requirements of Batch Gradient Descent (processing the entire dataset at once).

The choice of the batch size is a crucial hyperparameter. A small batch size introduces more noise into the parameter updates but can provide faster convergence, while a larger batch size may result in more stable updates but slower convergence and potentially increased memory requirements.

In practice, Mini-Batch Gradient Descent is widely used in training deep learning models due to its efficiency and balanced characteristics. The choice of batch size depends on factors such as the dataset size, available computational resources, and the model architecture.


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

## **Load a Data**

```python
from sklearn.datasets import load_diabetes
```

```python
# Read the Diabetes data
X, y = load_diabetes(return_X_y=True)
```

```python
X.shape
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape
```

## **Apply Linear Regression with Ordinary Least Squares (OLS)**

```python
from sklearn.linear_model import LinearRegression
```

```python
# Instantiate a 'LinearRegression' object
lr = LinearRegression()

# Fit the data
lr.fit(X_train, y_train)
```

```python
# Print the coefficients and intercept
print("Coefficients:\n", lr.coef_, "\n")
print("Intercept:", lr.intercept_)
```

```python
# Predict the test data
y_pred = lr.predict(X_test)
```

```python
# Calculate the R2 Score
from sklearn.metrics import r2_score
```

```python
print("R2 Score:", r2_score(y_test, y_pred))
```

## **Apply Multiple Linear Regression with Mini-Batch Gradient Descent**

```python
import random
```

```python
# Create a class to apply gradient descent
class MBGDRegressor:
    def __init__(self, batch_size, lr=0.01, epochs=100):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X_train, y_train):
        # Initialize the coefficients
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            
            for j in range(int(X_train.shape[0]/self.batch_size)):
                # Generate a list with random numbers 
                idx = random.sample(range(X_train.shape[0]), self.batch_size)
                
                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_
                
                # Update all the coefficients and intercept value
                intercept_der = -2 * np.mean(y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                
                coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)
                
    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
```

```python
# Instantiate a GDRegressor object
mbgdr = MBGDRegressor(batch_size=int(X_train.shape[0]/10), lr=0.01, epochs=50)

# Fit the data
mbgdr.fit(X_train, y_train)
```

```python
# Print the coefficients and intercept
print("Coefficients:\n", mbgdr.coef_, "\n")
print("Intercept:", mbgdr.intercept_)
```

```python
# Predict the test data
y_pred = mbgdr.predict(X_test)
```

```python
print("R2 Score:", r2_score(y_test, y_pred))
```

## **Mini-Batch Gradient Descent with Scikit-Learn**

```python
from sklearn.linear_model import SGDRegressor
```

```python
# Instantiate a SGDRegressor object
reg = SGDRegressor(learning_rate="constant", eta0=0.15)
```

```python
# Define the batch size and epochs
batch_size = 35
epochs = 100

for i in range(epochs):
    idx = random.sample(range(X_train.shape[0]), batch_size)
    reg.partial_fit(X_train[idx], y_train[idx])
```

```python
# Print the coefficients and intercept
print("Coefficients:\n", reg.coef_, "\n")
print("Intercept:", reg.intercept_)
```

```python
# Predict the test data
y_pred = reg.predict(X_test)
```

```python
print("R2 Score:", r2_score(y_test, y_pred))
```

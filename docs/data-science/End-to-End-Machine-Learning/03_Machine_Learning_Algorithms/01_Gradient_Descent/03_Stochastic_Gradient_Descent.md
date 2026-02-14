[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/01_Gradient_Descent/03_Stochastic_Gradient_Descent.ipynb)

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

# **Stochastic Gradient Descent**
Stochastic Gradient Descent (SGD) with a single-row update, also known as online SGD, is a variant of the traditional SGD algorithm where instead of using mini-batches of data for each parameter update, you update the model's parameters one data point (or row) at a time. In other words, after processing each individual data point, the model's parameters are updated based on the gradient computed for that single data point. This approach is sometimes referred to as "online learning."


<center><img src="https://editor.analyticsvidhya.com/uploads/58182variations_comparison.png" style="width: 60%"></ceneter>


Here's how SGD with a single-row update works:

1. **Initialization**: Start with an initial set of model parameters.

2. **Data Shuffling**: The training dataset is often shuffled to ensure that the order of data points does not bias the training process.

3. **Iterative Updates**: For each training iteration (or epoch), the algorithm processes one data point from the training set. The model's parameters are updated based on the gradient of the loss function with respect to that single data point.

4. **Gradient Computation**: The gradient of the loss function with respect to the model parameters is computed by backpropagating errors through the network (for neural networks) or using analytical derivatives (for simpler models). The gradient represents how the loss changes with small perturbations in the model parameters for that specific data point.

5. **Parameter Update**: The model parameters are adjusted in the opposite direction of the gradient, just like in traditional SGD. The learning rate controls the step size during each update.

6. **Repeat**: Steps 3-5 are repeated for the entire training dataset or until convergence criteria are met.

Online SGD can have some advantages and disadvantages:

**Advantages:**

1. **Efficiency**: Online SGD can be very efficient, as it processes one data point at a time, making it suitable for streaming data or scenarios with limited memory.

2. **Quick Convergence**: Online SGD can converge quickly, especially when the data is abundant and diverse.

3. **Adaptability**: It can adapt to changing data distributions and non-stationary data, making it useful in online learning and real-time applications.

**Disadvantages:**

1. **High Variability**: Since updates are based on individual data points, the parameter updates can be highly variable and noisy, which may result in a less stable convergence.

2. **Slower Convergence**: Online SGD can converge slower than traditional SGD with mini-batches due to the high variance in parameter updates.

3. **Difficulty in Hyperparameter Tuning**: Choosing an appropriate learning rate and other hyperparameters can be more challenging because of the high variance in updates.

Online SGD is typically used in situations where computational resources or memory are limited, or when the data distribution is constantly changing. It's commonly employed in online learning scenarios, such as recommendation systems, where new data arrives continuously and must be processed as it comes in. However, it may require careful tuning and monitoring to achieve optimal convergence and performance.


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
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
# Store the initial time in a variable
start = time.time()

# Instantiate a linear regression object
lr = LinearRegression()

# Fit the data
lr.fit(X_train, y_train)

# Print the actual time taken to fit the data
print("The Time taken is:", time.time() - start)
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

## **Apply Multiple Linear Regression with Stochastic Gradient Descent**

```python
# Create a class to apply gradient descent
class SGDRegressor:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                idx = np.random.randint(0, X_train.shape[0])
                
                # Predict the y_hat
                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_
                
                # Update the intercept using a single row
                intercept_der = -2 * (y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                
                # Update the coefficients using a single row
                coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)
    
    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
```

```python
# Store the initial time in a variable
start = time.time()

# Instantiate a SGDRegressor object
sgdr = SGDRegressor(lr=0.03, epochs=100)

# Fit the data
sgdr.fit(X_train, y_train)

# Print the actual time taken to fit the data
print("The Time taken is:", time.time() - start)
```

```python
# Print the coefficients and intercept
print("Coefficients:\n", sgdr.coef_, "\n")
print("Intercept:", sgdr.intercept_)
```

```python
# Predict the test data
y_pred = sgdr.predict(X_test)
```

```python
# Calculate the R2 Score
print("R2 Score:", r2_score(y_test, y_pred))
```

## **Stochastic Gradient Descent with Scikit-Learn**

```python
from sklearn.linear_model import SGDRegressor
```

```python
# Instantiate a SGDRegressor object
reg = SGDRegressor(loss='squared_error', learning_rate='constant', eta0=0.01)

# Fit the data
reg.fit(X_train, y_train)
```

```python
# Predict the test data
y_pred = reg.predict(X_test)
```

```python
# Calculate the R2 Score
print("R2 Score:", r2_score(y_test, y_pred))
```

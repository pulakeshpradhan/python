[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/01_Gradient_Descent/02_Batch_Gradient_Descent.ipynb)

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

# **Batch Gradient Descent**
Batch Gradient Descent is a popular optimization algorithm used in machine learning and deep learning for training models, particularly for supervised learning tasks like linear regression and neural network training. It's a type of gradient descent algorithm that updates the model's parameters based on the average gradient of the loss function with respect to the **entire training dataset**.


<center><img src="https://editor.analyticsvidhya.com/uploads/58182variations_comparison.png" style="width: 60%"></ceneter>


Here's how Batch Gradient Descent works:

1. **Initialization**: Initialize the model parameters randomly or with some predefined values.

2. **Batch Selection**: Divide the training dataset into smaller subsets called batches. Each batch contains a fixed number of training examples. The batch size is a hyperparameter that you can choose, and it determines how many examples are used in each parameter update.

3. **Compute Gradient**: For each batch, compute the gradient of the loss function with respect to the model parameters. This gradient represents the direction and magnitude of the steepest ascent of the loss function for that batch.

4. **Update Parameters**: Update the model parameters by moving in the opposite direction of the gradient. The update rule typically follows this formula:

   ```
   θ_new = θ_old - learning_rate * (∇(Loss) / ∇(θ))
   ```

   where:
   - θ_new is the updated parameter values.
   - θ_old is the current parameter values.
   - learning_rate is a hyperparameter that controls the step size or learning rate of the optimization.
   - ∇(Loss) represents the gradient of the loss function.
   - ∇(θ) represents the gradient with respect to the model parameters.

5. **Repeat**: Repeat steps 3 and 4 for all the batches in the training dataset. This constitutes one epoch.

6. **Convergence Check**: Monitor the convergence of the algorithm by checking if the loss function has sufficiently decreased or other convergence criteria are met. If not, repeat steps 3 to 5 for more epochs.

7. **Termination**: Stop the training process when a stopping criterion is met. This could be a maximum number of epochs, a certain level of loss convergence, or other criteria.

Batch Gradient Descent has some advantages, such as stable and consistent updates, but it can be computationally expensive, especially when working with large datasets. It also requires the entire dataset to fit in memory, which may not be feasible for very large datasets. In such cases, variants of gradient descent, like Stochastic Gradient Descent (SGD) and Mini-Batch Gradient Descent, are often used to overcome these limitations.


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
X.shape, y.shape
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
# Instantiate a linear regression object
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

## **Apply Multiple Linear Regression with Batch Gradient Descent**

```python
# Create a class to apply gradient descent
class GDRegressor:
    # Constructor
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            # Update all the coefficients and intercept value
            y_hat = np.dot(X_train, self.coef_) + self.intercept_
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            
            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)
              
    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_ 
```

```python
# Instantiate a GDRegressor object
gdr = GDRegressor(lr=0.3, epochs=1000)

# Fit the data
gdr.fit(X_train, y_train)
```

```python
# Print the coefficients and intercept
print("Coefficients:\n", gdr.coef_, "\n")
print("Intercept:", gdr.intercept_)
```

```python
# Predict the test data
y_pred = gdr.predict(X_test)
```

```python
# Calculate the R2 Score
from sklearn.metrics import r2_score

print("R2 Score:", r2_score(y_test, y_pred))
```

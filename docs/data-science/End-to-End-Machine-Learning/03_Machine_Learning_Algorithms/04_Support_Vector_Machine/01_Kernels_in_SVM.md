[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/03_Machine_Learning_Algorithms/04_Support_Vector_Machine/01_Kernels_in_SVM.ipynb)

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

# **Kernels in SVM**
In Support Vector Machines (SVM), kernels play a crucial role in extending the algorithm's capability to handle non-linearly separable data. Kernels enable SVM to implicitly map the input data into a higher-dimensional feature space where a linear decision boundary can be found. This process is known as the "kernel trick," and it allows SVM to effectively handle complex relationships between features without explicitly computing the transformation.

Here are some common types of kernels used in SVM:

1. **Linear Kernel:**
   The linear kernel is the simplest kernel, and it computes the dot product between the feature vectors in the original input space. It is suitable for linearly separable data or cases where a linear decision boundary is appropriate.

2. **Polynomial Kernel:**
   The polynomial kernel computes the similarity between two vectors as the polynomial of their dot product. It introduces additional parameters such as the degree of the polynomial and a coefficient. It is useful for capturing non-linear relationships between features.

3. **Radial Basis Function (RBF) Kernel:**
   The RBF kernel, also known as the Gaussian kernel, computes the similarity between two vectors based on the radial distance between them. It is defined by a single parameter called the gamma (γ) parameter. The RBF kernel is highly flexible and can capture complex non-linear decision boundaries.

4. **Sigmoid Kernel:**
   The sigmoid kernel computes the similarity between two vectors using a sigmoid function. It introduces additional parameters such as the slope and the intercept. The sigmoid kernel can be useful for modeling non-linear relationships, but it is less commonly used compared to linear and RBF kernels.

5. **Custom Kernels:**
   In addition to the standard kernels mentioned above, SVM also allows for the use of custom kernels. Custom kernels can be defined based on domain knowledge or specific problem characteristics, allowing for more flexibility in modeling complex relationships in the data.

The choice of kernel in SVM depends on the characteristics of the data and the problem at hand. Experimentation and cross-validation are often used to determine the most appropriate kernel for a given task. Additionally, tuning kernel parameters, such as the degree of the polynomial or the gamma parameter in the RBF kernel, can significantly impact the performance of the SVM model.


<center><img src="https://miro.medium.com/v2/resize:fit:786/format:webp/1*_Uhpj662QpxoIa8qlPYJ9A.png" width="50%"></center>
<center><img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_001.png" width="50%"></center>


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import plotly.express as px

sns.set_style("darkgrid")
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

import warnings
warnings.filterwarnings("ignore")
```

## **Make a Circular Dataset**

```python
# Generate a cicular dataset
X, y = make_circles(n_samples=200, noise=0.1, factor=0.2, random_state=0)

# Plot the Data
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, edgecolor="black", linewidth=0.5);
```

## **Train Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train.shape, X_test.shape
```

## **Application of SVC with Different Kernels**


### **SVC with Linear Kernel**

```python
# Instantiate a SVC classifier object with 'linear' kernel
svc = SVC(kernel="linear", C=1.0)

# Fit the training data
svc.fit(X_train, y_train)
```

```python
# Predict the test data
y_pred = svc.predict(X_test)

# Check the accuracy
print("Accuracy of SVC with Linear Kernel:", accuracy_score(y_test, y_pred).round(4))
```

```python
# Write a function to plot the decision boundary
def plot_decision_boundary(X, y, kernel="linear", C=1.0, degree=1):
    '''Plot the decision boundary for a 2D SVC'''
    
    # Train the SVC model
    svc = SVC(kernel=kernel, C=C, degree=degree)
    
    # Fit the training data
    svc.fit(X, y)
    
    # Create a mesh grid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict the labels for all points in the mesh grid
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and the data points
    plt.contourf(xx, yy, Z, cmap="RdYlGn", alpha=0.5)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=["red", "green"], edgecolors='k', linewidth=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f"Decision boundary of SVC with '{kernel}' kernel")
    plt.show()
```

```python
plot_decision_boundary(X, y, kernel="linear", C=1.0)
```

### **SVC with RBF Kernel**
The Radial Basis Function (RBF) kernel, also known as the Gaussian kernel, is a popular choice in Support Vector Machine (SVM) algorithms, particularly for handling non-linearly separable data. It measures the similarity between two data points in a high-dimensional space and is defined by a single parameter called the gamma (γ) parameter.

#### **Mathematical Formulation:**

The RBF kernel $ K(\mathbf{x}_i, \mathbf{x}_j) $ between two feature vectors $ \mathbf{x}_i $ and $ \mathbf{x}_j $ is computed as:

$$ K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2\right) $$

Here:
- $ ||\mathbf{x}_i - \mathbf{x}_j||^2 $ represents the squared Euclidean distance between the two feature vectors.
- $ \gamma $ (gamma) is a hyperparameter that controls the influence of each training example on the decision boundary. It determines the "spread" of the kernel function.

#### **Characteristics:**

1. **Flexibility:** The RBF kernel is highly flexible and can capture complex non-linear decision boundaries in the data.
   
2. **Implicit Mapping:** It implicitly maps the input data into a higher-dimensional feature space, where a linear decision boundary can be found, even if the original data is not linearly separable.

3. **Smoothness:** The RBF kernel produces smooth decision boundaries, which can be advantageous in many real-world classification tasks.

4. **Hyperparameter Sensitivity:** The performance of the RBF kernel is sensitive to the choice of the gamma parameter. Higher values of gamma lead to more complex decision boundaries and may result in overfitting, while lower values of gamma lead to smoother decision boundaries and may result in underfitting.

#### **Applications:**

- **Non-linear Classification:** The RBF kernel is commonly used in SVM for tasks involving non-linearly separable data, such as image classification, text classification, and bioinformatics.
  
- **Clustering:** The RBF kernel can also be used in clustering algorithms, such as the Gaussian Radial Basis Function (RBF) kernel in K-means clustering.

In summary, the RBF kernel is a powerful tool in SVM algorithms for handling non-linear relationships in the data and is widely used in various machine learning applications due to its flexibility and effectiveness. However, careful tuning of the gamma parameter is essential to achieve optimal performance and avoid overfitting or underfitting.

```python
# Write a function to apply RBF on the data
def applyRBF(X, y, gamma=1, X_center=0):
    r = np.exp(-gamma*((X - X_center) ** 2)).sum(axis=1)
    
    fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=r, color=y.astype("str"))
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color="black")))
    fig.update_layout(width=800, height=600)
    fig.show()
    
applyRBF(X, y, gamma=1, X_center=0)
```

```python
# Instantiate a SVC classifier object with 'rbf' kernel
svc = SVC(kernel="rbf", C=1.0)

# Fit the training data
svc.fit(X_train, y_train)
```

```python
# Predict the test data
y_pred = svc.predict(X_test)

# Check the accuracy
print("Accuracy of SVC with RBF Kernel:", accuracy_score(y_test, y_pred).round(4))
```

```python
plot_decision_boundary(X, y, kernel="rbf", C=1.0)
```

### **SVC with Polynomial Kernel**
The Polynomial Kernel is a kernel function commonly used in Support Vector Machines (SVM) for handling non-linear relationships in the data. It maps the input data into a higher-dimensional space using polynomial functions, allowing SVM to find non-linear decision boundaries. The Polynomial Kernel is defined by a degree parameter, which determines the degree of the polynomial used for the mapping.

#### **Mathematical Formulation:**

The Polynomial Kernel $ K(\mathbf{x}_i, \mathbf{x}_j) $ between two feature vectors $ \mathbf{x}_i $ and $ \mathbf{x}_j $ is computed as:

$$ K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d $$

Here:
- $ \mathbf{x}_i^T \mathbf{x}_j $ represents the dot product between the two feature vectors.
- $ d $ is the degree of the polynomial, which determines the complexity of the decision boundary.
- $ c $ is an optional parameter known as the bias or offset, which can be used to control the influence of lower-degree polynomial terms.

#### **Characteristics:**

1. **Non-Linearity:** The Polynomial Kernel allows SVM to capture non-linear relationships between features by mapping them into a higher-dimensional space.

2. **Flexibility:** The degree parameter allows for varying levels of complexity in the decision boundary. Higher degree polynomials can capture more complex relationships but may also lead to overfitting.

3. **Computationally Efficient:** While the Polynomial Kernel increases the dimensionality of the feature space, it is computationally more efficient compared to other non-linear kernels like the Gaussian Radial Basis Function (RBF) kernel.

#### **Applications:**

- **Non-linear Classification:** The Polynomial Kernel is commonly used in SVM for tasks involving non-linearly separable data, such as image classification, text classification, and pattern recognition.
  
- **Feature Engineering:** The Polynomial Kernel can be used as a form of feature engineering to transform the input features into a higher-dimensional space, where linear separation may be easier to achieve.

In summary, the Polynomial Kernel is a powerful tool in SVM algorithms for handling non-linear relationships in the data. By adjusting the degree parameter, practitioners can control the complexity of the decision boundary and tailor the SVM model to the specific characteristics of the data.

```python
# Instantiate a SVC classifier object with 'poly' kernel
svc = SVC(kernel="poly", C=1.0, degree=2)

# Fit the training data
svc.fit(X_train, y_train)
```

```python
# Predict the test data
y_pred = svc.predict(X_test)

# Check the accuracy
print("Accuracy of SVC with Polynomial Kernel:", accuracy_score(y_test, y_pred).round(4))
```

```python
plot_decision_boundary(X, y, kernel="poly", C=1.0, degree=2)
```

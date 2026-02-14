[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/pytorch-for-deep-learning-and-machine-learning/PyTorch_CampusX/04_pytorch_training_pipeline.ipynb)

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

# **PyTorch Training Pipeline**


## **Import Dependencies**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchmetrics import Accuracy

import warnings
warnings.filterwarnings('ignore')
```

## **Read the Dataset**

```python
# Load the breast cancer dataset using Pandas
data = pd.read_csv(r"D:\GITHUB\PyTorch-for-Deep-Learning-and-Machine-Learning\datasets\breast_cancer_data.csv")
print(data.shape)
data.head()
```

## **Data Pre-processing**


### **Data Cleaning**

```python
# Drop the irrelevant columns
data.drop(columns=['id', 'Unnamed: 32'], inplace=True)
print(data.shape)
data.head()
```

### **Train-Test Split**

```python
# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns='diagnosis'),
    data['diagnosis'],
    test_size=0.3,
    random_state=42
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

### **Feature Scaling**

```python
# Print the column information
X_train.info()
```

```python
# Scale the input variables using standarad scaler
scaler = StandardScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape, X_test_scaled.shape)
```

### **Label Encoding**

```python
# Encode the target variable using label encoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(y_train_encoded.shape, y_test_encoded.shape)
```

### **Convert NumPy Arrays to PyToch Tensors**

```python
X_train_tensor = torch.from_numpy(X_train_scaled)
X_test_tensor = torch.from_numpy(X_test_scaled)
y_train_tensor = torch.from_numpy(y_train_encoded)
y_test_tensor = torch.from_numpy(y_test_encoded)

print(X_train_tensor.shape, X_test_tensor.shape, y_train_encoded.shape, y_test_encoded.shape)
```

## **Building the Model**

```python
class MySimpleNN:

    def __init__(self, X: torch.Tensor):

        self.weights = torch.rand(X.shape[1], 1, dtype=torch.float64, requires_grad=True)
        self.bias = torch.rand(1, dtype=torch.float64, requires_grad=True)

    def forward(self, X: torch.Tensor):
        
        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)

        return y_pred
    
    def loss_func(self, y_pred, y):
        # Clamp predictions to avoid log(0)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

        # Calculate loss
        loss = -(y_train_tensor * torch.log(y_pred) + (1 - y_train_tensor) * torch.log(1 - y_pred)).mean()
        
        return loss
```

### **Model Parameters**

```python
# Set the learning rate and number of epoch
lr = 0.1 # learning rate
epochs = 25
```

## **Training Pipeline**

```python
# Create an object of the model
model = MySimpleNN(X_train_tensor)

# Define a loop
for epoch in range(epochs):

    # Forward pass
    y_pred = model.forward(X_train_tensor)

    # Loss calculation
    loss = model.loss_func(y_pred, y_train_tensor)

    # Backward pass
    loss.backward()

    # Parameters update
    with torch.no_grad():
        model.weights -= lr * model.weights.grad
        model.bias -= lr * model.bias.grad

    # Zero gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    # Print loss in each epoch
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
```

## **Model Evaluation**

```python
# Model evaluation
with torch.no_grad():
    y_pred = model.forward(X_test_tensor)
    y_pred = (y_pred > 0.5).float()

# Calculate the accuracy using torch metrics
accuracy = Accuracy(task='binary')
print('Accuracy on testing data:', accuracy(y_pred.squeeze(), y_test_tensor).item())
```

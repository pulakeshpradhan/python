[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/pytorch-for-deep-learning-and-machine-learning/PyTorch_CampusX/05_pytorch_nn_module.ipynb)

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

# **PyTorch NN Module**

The torch.nn module in PyTorch is a core library that provides a wide array of classes and
functions designed to help developers build neural networks efficiently and effectively. It
abstracts the complexity of creating and training neural networks by offering pre-built layers,
loss functions, activation functions, and other utilities, enabling you to focus on designing and
experimenting with model architectures.

**Key Components of torch.nn:**
1. Modules (Layers):
    - `nn.Module`: The base class for all neural network modules. Your custom models and
    layers should subclass this class.
    - Common Layers: Includes layers like `nn.Linear` (fully connected layer), `nn.Conv2d`
    (convolutional layer), `nn.LSTM` (recurrent layer), and many others.

2. Activation Functions:
    - Functions like `nn.ReLU`, `nn.Sigmoid`, and `nn.Tanh` introduce non-linearities to the model, allowing it to learn complex patterns.

3. Loss Functions:
    - Provides loss functions such as `nn.CrossEntropyLoss`, `nn.MSELoss`, and `nn.NLLLoss` to quantify the difference between the model's predictions and the actual targets.

4. Container Modules:
    - `nn.Sequential`: A sequential container to stack layers in order.

5. Regularization and Dropout:
    - Layers like `nn.Dropout` and `nn.BatchNorm2d` help prevent overfitting and improve the model's ability to generalize to new data.


## **Import Dependencies**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch import nn
from torchinfo import summary
from torchmetrics import Accuracy

import warnings
warnings.filterwarnings('ignore')
```

## **Read the Dataset**

```python
# Load the breast cancer dataset using Pandas
data = pd.read_csv(r"D:\GITHUB\pytorch-for-deep-Learning-and-machine-learning\datasets\breast_cancer_data.csv")
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
    data.drop(columns=['diagnosis']),
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

### **Convert NumPy Arrays to PyTorch Tensor**

```python
X_train_tensor = torch.from_numpy(X_train_scaled).type(torch.float32)
X_test_tensor = torch.from_numpy(X_test_scaled).type(torch.float32)
y_train_tensor = torch.from_numpy(y_train_encoded).type(torch.float32)
y_test_tensor = torch.from_numpy(y_test_encoded).type(torch.float32)

print(X_train_tensor.shape, X_test_tensor.shape, y_train_encoded.shape, y_test_encoded.shape)
```

## **Build a Simple NN Model**

```python
# Create a simple neural network model with a single node
class MySimpleNN(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor):
        out = self.linear(X)
        out = self.sigmoid(out)

        return out
```

### **Training Pipeline**

```python
# Create an object of the model
model = MySimpleNN(X_train_tensor.shape[1])
# Show the model summary
summary(model)
```

```python
# Set the learning rate and number of epoch
lr = 0.1 # learning rate
epochs = 25

# Define a loss function and an optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

```python
# Define a loop
for epoch in range(epochs):

    # Forward pass
    y_pred = model(X_train_tensor)

    # Loss calculation
    loss = loss_fn(y_pred.squeeze(), y_train_tensor)

    # Zero gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Parameters update
    optimizer.step()

    # Print loss in epoch
    print(f'Epoch: {epoch + 1}, Loss: {loss}')
```

```python
# Print the model weights and bias
print('Model weights:')
print(model.linear.weight)

print('Model bias:')
print(model.linear.bias)
```

### **Model Evaluation**

```python
# Make predictions on testing data
with torch.no_grad():
    y_pred = model(X_test_tensor)

# Calculate the accuracy using torchmetrics
accuracy = Accuracy(task='binary')
print('Accuracy on testing data:', accuracy(y_pred.squeeze(), y_test_tensor).item())
```

## **Build a NN Model with a Hidden Layer**

```python
# Create a neural network with a hidden layers
class MyComplexNN(nn.Module):

    def __init__(self, num_feature):
        super().__init__()
        self.linear1 = nn.Linear(in_features=num_feature, out_features=3)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=3, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor):
        out = self.linear1(X)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        return out
```

```python
# Build the same model with sequential container
class MyComplexNN(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 3), # input layer
            nn.ReLU(), # activation
            nn.Linear(3, 1), # hidden layer
            nn.Sigmoid()
        )

    def forward(self, X: torch.Tensor):
        out = self.network(X)

        return out
```

### **Training Pipeline**

```python
# Create an object of the model class
model = MyComplexNN(X_train_tensor.shape[1])
# Print the model summary
summary(model)
```

```python
# # Set the learning rate and number of epoch
lr = 0.01
epochs = 25

# Define a loss function and an optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

```python
# Define a loop
for epoch in range(epochs):

    # Forward pass
    y_pred = model(X_train_tensor)

    # Loss calculation
    loss = loss_fn(y_pred.squeeze(), y_train_tensor)

    # Zero gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Parameters update
    optimizer.step()

    # Print epoch loss
    print(f'Epoch: {epoch + 1}, Loss: {loss}')
```

```python
# Print the model weights and biases
print('Model weights:')
print(model.network[0].weight)
print('Model biases:')
print(model.network[0].bias)
```

### **Model Evaluation**

```python
# Make predictions on testing data
with torch.no_grad():
    y_pred = model(X_test_tensor)

# Calculate the accuracy using torchmetrics
accuracy = Accuracy(task='binary')
print('Accuracy on testing data:', accuracy(y_pred.squeeze(), y_test_tensor).item())
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/pytorch-for-deep-learning-and-machine-learning/PyTorch_CampusX/07_buidling_ann_using_pytorch.ipynb)

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

# **Building ANN using PyTorch**


## **Import Dependencies**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torchmetrics.classification import MulticlassAccuracy

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = 'Times New Roman'

# Set random seeds for reproducibility
torch.manual_seed(42)

# Setup device agnostic code to run the model on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Current device: {device}')
```

## **Read the Dataset**

```python
# Load the Fashion MNIST dataset using Pandas
train_data = pd.read_csv(r'D:\GITHUB\pytorch-for-deep-learning-and-machine-learning\datasets\fashion-mnist_train.csv')
test_data = pd.read_csv(r'D:\GITHUB\pytorch-for-deep-learning-and-machine-learning\datasets\fashion-mnist_test.csv')

print(train_data.shape, test_data.shape)
train_data.head()
```

## **Data Pre-processing**


### **Split the Data into Input and Target**

```python
# Split the data into target and labels
X_train, y_train = train_data.drop(columns=['label']), train_data['label']
X_test, y_test = test_data.drop(columns=['label']), test_data['label']

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train.head()
```

```python
# Create a 4x4 grid of random images from training data
random_ids = [random.randint(0, X_train.shape[0]) for i in range(16)]

fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    img = X_train.iloc[random_ids[i]].values.reshape(28, 28)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Label: {y_train[random_ids[i]]}')

plt.tight_layout()
plt.show()
```

### **Scale the Data**

```python
X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape, X_test.shape)
```

### **Convert NumPy Arrays to PyTorch Tensors**

```python
X_train = torch.from_numpy(X_train.values).type(torch.float32)
X_test = torch.from_numpy(X_test.values).type(torch.float32)
y_train =  torch.from_numpy(y_train.values).type(torch.long)
y_test =  torch.from_numpy(y_test.values).type(torch.long)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

## **Create Dataset and DataLoader**

```python
# Create a Fashion MNIST dataset class
class FashionMNISTDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

# Create the training and testing dataset
train_dataset = FashionMNISTDataset(X_train, y_train)
test_dataset = FashionMNISTDataset(X_test, y_test)
```

```python
# Create dataloader object
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, pin_memory=True)
```

## **Build an Artificial Neural Network (ANN) Model**

```python
# Create an ANN class with two hidden layers
class MyANNModel(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, X: torch.Tensor):
        out = self.network(X)
        return out
```

### **Training Pipeline**

```python
# Create an object of the model
model = MyANNModel(X_train.shape[1]).to(device)
summary(model)
```

```python
# Define the learning rate and the number of epochs
lr = 0.1
epochs = 50

# Define the optimizer and the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
```

```python
# Define a loop 
for epoch in range(epochs):

    ## Training
    model.train()

    total_train_loss = 0

    for batch_X, batch_y in train_dataloader:

        # Move data to gpu
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        y_pred = model(batch_X)

        # Loss calculation
        loss = loss_fn(y_pred, batch_y)
        total_train_loss += loss

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

    ## Testing
    model.eval()

    total_test_loss = 0

    for batch_X, batch_y in test_dataloader:

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        with torch.no_grad():
            test_pred = model(batch_X)
            loss = loss_fn(test_pred, batch_y)
            total_test_loss += loss

    # Print the epoch training and testing loss
    print(f'Epoch: {epoch+1}, Train Loss: {(total_train_loss / len(train_dataloader)):.4f}, Test Loss: {(total_test_loss / len(test_dataloader)):.2f}')
```

## **Model Evaluation**

```python
# Model evaluation using test dataloader
model.eval()

# Initialize the accuracy metric
accuracy = MulticlassAccuracy(num_classes=10).to(device)  # Move the metric to the same device as the model

# Evaluate on the test dataset
for batch_X, batch_y in test_dataloader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    # Forward pass
    y_pred = model(batch_X)
    y_pred = torch.argmax(y_pred, axis=1)  # Get the predicted class indices

    # Update the accuracy metric
    accuracy.update(y_pred, batch_y)

# Compute the final accuracy
final_accuracy = accuracy.compute()
print(f'Accuracy: {final_accuracy:.2f}')
```

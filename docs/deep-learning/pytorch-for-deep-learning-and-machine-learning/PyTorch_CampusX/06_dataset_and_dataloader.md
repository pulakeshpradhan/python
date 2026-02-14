[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/pytorch-for-deep-learning-and-machine-learning/PyTorch_CampusX/06_dataset_and_dataloader.ipynb)

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

# **Datasets and DataLoaders**
In PyTorch, `Dataset` and `DataLoader` are fundamental for handling data. Here's a breakdown of how they work and how to use them:

**1. Dataset**
The Dataset class is essentially a blueprint. When you create a custom Dataset, you decide how data is loaded and returned. It defines:
- `__init__()`: which tells how data should be loaded.
- `__len__()`: which returns the total number of samples.
- `__getitem__(index)`: which returns the data (and label) at the given index.

**2. DataLoader**
The DataLoader wraps a Dataset and handles batching, shuffling, and parallel loading for you.

DataLoader Control Flow:
- At the start of each epoch, the DataLoader (if shuffle=True) shuffles indices(using a sampler).
- It divides the indices into chunks of batch_size.
- for each index in the chunk, data samples are fetched from the Dataset object
- The samples are then collected and combined into a batch (using collate_fn)
- The batch is returned to the main training loop

**Tips:**
1. **Custom Collate Function**: For datasets that have variable-length inputs, you can use a custom collate function.
2. **Lazy Loading**: If your dataset is too large, implement lazy loading in the `__getitem__` method by loading data directly from files.
3. **Data Augmentation**: Use `torchvision.transforms` for on-the-fly data augmentation.


#### **A Note about Samplers**
In PyTorch, the sampler in the DataLoader determines the strategy for selecting samples from the dataset during data loading. It controls how indices of the dataset are drawn for each batch.

**Types of Samplers:**
PyTorch provides several predefined samplers, and you can create custom ones:
1. `SequentialSampler`:
   - Samples elements sequentially, in the order they appear in the dataset.
   - Default when shuffle=False.
2. `RandomSampler`:
   - Samples elements randomly without


#### **A Note about `collate_fn`**
The collate_fn in PyTorch's DataLoader is a function that specifies how to combine a list of samples from a dataset into a single batch. By default, the DataLoader uses a simple batch collation mechanism, but collate_fn allows you to customize how the data should be processed and batched.


#### **Important Parameters of DataLoader**
The DataLoader class in PyTorch comes with several parameters that allow you to customize how data is loaded, batched, and preprocessed. Some of the most commonly used and important parameters include:

1. **dataset (mandatory)**:
    - The Dataset from which the DataLoader will pull data.
    - Must be a subclass of torch.utils.data.Dataset that implements `__getitem__` and `__len__`.

2. **batch_size**:
    - How many samples per batch to load.
    - Default is 1.
    - Larger batch sizes can speed up training on GPUs but require more memory.

3. **shuffle**:
    - If True, the DataLoader will shuffle the dataset indices each epoch.
    - Helpful to avoid the model becoming too dependent on the order of samples.

4. **num_workers**:
    - The number of worker processes used to load data in parallel.
    - Setting num_workers > 0 can speed up data loading by leveraging multiple CPU cores, especially if I/O or preprocessing is a bottleneck.

5. **pin_memory**:
    - If True, the DataLoader will copy tensors into pinned (page-locked) memory before returning them.
    - This can improve GPU transfer speed and thus overall training throughput, particularly on CUDA systems.

6. **drop_last**:
    - If True, the DataLoader will drop the last incomplete batch if the total number of samples is not divisible by the batch size.
    - Useful when exact batch sizes are required (for example, in some batch normalization scenarios).

7. **collate_fn**:
    - A callable that processes a list of samples into a batch (the default simply stacks tensors).
    - Custom collate_fn can handle variable-length sequences, perform custom batching logic, or handle complex data structures.

8. **sampler**:
    - sampler defines the strategy for drawing samples (e.g., for handling imbalanced classes, or custom sampling strategies).
    - batch_sampler works at the batch level, controlling how batches are formed.
    - Typically, you don’t need to specify these if you are using batch_size and shuffle. However, they provide lower-level control if you have advanced requirements.


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
from torch.utils.data import Dataset, DataLoader
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

### **Convert NumPy Arrays to PyTorch Tensors**

```python
X_train_tensor = torch.from_numpy(X_train_scaled).type(torch.float32)
X_test_tensor = torch.from_numpy(X_test_scaled).type(torch.float32)
y_train_tensor = torch.from_numpy(y_train_encoded).type(torch.float32)
y_test_tensor = torch.from_numpy(y_test_encoded).type(torch.float32)

print(X_train_tensor.shape, X_test_tensor.shape, y_train_encoded.shape, y_test_encoded.shape)
```

## **Create Dataset and DataLoader**

```python
# Create a custon dataset class
class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
```

```python
# Create an object of the custom dataset class
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)

print('Length of the training dataset:', train_dataset.__len__())
# Print a row from train dataset
train_dataset.__getitem__(10)
```

```python
# Create dataloader object
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Print first two batches from the training dataaset
for idx , (batch_features, batch_labels) in enumerate(train_dataloader):
    print(batch_features)
    print(batch_labels)
    print('-'*50)

    if idx == 1:
        break
```

## **Build a Neural Network Model**

```python
class MyModel(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, X: torch.Tensor):
        out = self.network(X)
        return out
```

### **Training Pipeline**

```python
# Create an object of the model
model = MyModel(X_train_tensor.shape[1])
summary(model)
```

```python
# Set the learning rate and number of epochs
lr = 0.01
epochs = 50

# Define a loss function and an optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
```

```python
# Define a loop
for epoch in range(epochs):

    avg_loss = 0

    # Iterate training batches
    for batch_X, batch_y in train_dataloader:

        # Forward pass
        y_pred = model(batch_X)

        # Loss calculation
        loss = loss_fn(y_pred.squeeze(), batch_y)
        avg_loss += loss

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

    # Validate on testing batches
    avg_test_loss = 0

    for batch_X, batch_y in test_dataloader:
        with torch.no_grad():
            test_preds = model(batch_X)
            test_loss = loss_fn(test_preds.squeeze(), batch_y)
            avg_test_loss += test_loss
            
    # Print the epoch loss
    print(f'Epoch: {epoch}, Loss: {(avg_loss / len(train_dataloader)):.2f}, Val Loss: {(avg_test_loss / len(test_dataloader)):.2f}')
```

## **Model Evaluation**

```python
# Model evaluation using test dataloader
model.eval() # Set the model to evaluation mode
accuracy_list = []
accuracy = Accuracy(task='binary')

# Make predictions on testing data
with torch.no_grad():
    for batch_X, batch_y in test_dataloader:
        y_pred = model(batch_X).squeeze()
        y_pred = (y_pred > 0.5).float()
        batch_acc = accuracy(y_pred, batch_y)

        accuracy_list.append(batch_acc)
        
# Calculate overall accuracy
overall_acc = np.mean(accuracy_list)
print(f'Accuracy: {overall_acc:.2f}')
```

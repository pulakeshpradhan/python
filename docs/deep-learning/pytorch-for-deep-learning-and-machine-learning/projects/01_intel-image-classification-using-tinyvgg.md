[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/pytorch-for-deep-learning-and-machine-learning/projects/01_intel-image-classification-using-tinyvgg.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# **Intel Image Classification using TinyVGG**


## **Import Dependencies**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
from tqdm import tqdm
import cv2
import random
import torch
from torch import nn
from torchmetrics import Accuracy
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Times New Roman"

torch.__version__
```

## **Explore the Data Folders**

```python
# Define the folder paths
train_path = "/kaggle/input/intel-image-classification/seg_train/seg_train"
test_path = "/kaggle/input/intel-image-classification/seg_test/seg_test"
pred_path = "/kaggle/input/intel-image-classification/seg_pred/seg_pred"

# Get the number of images per class for each paths
for path in [train_path, test_path]:
    print(f"{path}\n---------------")
    categorical_paths = glob(pathname=path+"/*")
    for cat_path in categorical_paths:
        folder_name = cat_path.split("/")[-1]
        no_of_images = len(os.listdir(cat_path))
        print(f"Found {no_of_images} images in folder {folder_name}.") 
        
    print("\n")
    
# Print the number of images in the 'seg_pred' folder
print(f"Found {len(os.listdir(pred_path))} images in folder seg_pred.")
```

## **Prepare the Data**

```python
# Check the size of the images
# Get the number of images per class for each paths
for path in [train_path, test_path]:
    print(f"{path}\n---------------")
    categorical_paths = glob(pathname=path+"/*")
    image_sizes = []
    for cat_path in tqdm(categorical_paths):
        file_paths = glob(cat_path+"/*")
        for f_path in file_paths:
            img = plt.imread(f_path)
            shape = img.shape
            image_sizes.append(shape)
            
    print(pd.Series(image_sizes).value_counts())
```

```python
# Resize the images and store in seperate variables
X_train, X_test = [], []
y_train, y_test = [], []
X_pred = []

code = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}

# Set the resize parameter
SIZE = 128

# Get the number of images per class for each paths
for path in [train_path, test_path]:
    print(f"{path}\n---------------")
    categorical_paths = glob(pathname=path+"/*")
    for cat_path in tqdm(categorical_paths):
        cat_name = cat_path.split("/")[-1]
        file_paths = glob(cat_path+"/*")
        for f_path in file_paths:
            img = cv2.imread(f_path)
            img_resized = cv2.resize(img, (SIZE, SIZE))
            
            if "train" in path:
                X_train.append(img_resized)
                y_train.append(code[cat_name])
            else:
                X_test.append(img_resized)
                y_test.append(code[cat_name])
                
for f_path in tqdm(glob(pathname=pred_path+"/*")):
    img = cv2.imread(f_path)
    img_resized = cv2.resize(img, (SIZE, SIZE))
    X_pred.append(img_resized)
```

```python
# Change the list to numpy array
X_train, X_test, X_pred = np.array(X_train), np.array(X_test), np.array(X_pred)
y_train, y_test = np.array(y_train), np.array(y_test)

X_train.shape, X_test.shape, X_pred.shape, y_train.shape, y_test.shape
```

```python
# Convert the channel last to channel first
X_train, X_test, X_pred = np.transpose(X_train, (0, 3, 1, 2)), np.transpose(X_test, (0, 3, 1, 2)), np.transpose(X_pred, (0, 3, 1, 2))
X_train.shape, X_test.shape, X_pred.shape, y_train.shape, y_test.shape
```

```python
# Plot some random images from the training data
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_id = random.randint(0, X_train.shape[0])
    X, y = X_train[random_id], y_train[random_id]
    fig.add_subplot(rows, cols, i)
    plt.imshow(np.transpose(X, [1, 2, 0]))
    plt.title([key for key, value in code.items() if value == y][0])
    plt.axis(False)
```

```python
# Plot some random images from the testing data
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_id = random.randint(0, X_test.shape[0])
    X, y = X_test[random_id], y_test[random_id]
    fig.add_subplot(rows, cols, i)
    plt.imshow(np.transpose(X, [1, 2, 0]))
    plt.title([key for key, value in code.items() if value == y][0])
    plt.axis(False)
```

```python
# Convert the data into tensors
X_train, y_train = torch.from_numpy(X_train).type(torch.float), torch.from_numpy(y_train).type(torch.LongTensor)
X_test, y_test = torch.from_numpy(X_test).type(torch.float), torch.from_numpy(y_test).type(torch.LongTensor)
X_pred = torch.from_numpy(X_pred).type(torch.float)
```

```python
# Apply normalization
X_train, X_test, X_pred = X_train/255, X_test/255, X_pred/255
X_train.shape, X_test.shape, X_pred.shape, y_train.shape, y_test.shape
```

## **Build the Model (TinyVGG) using PyTorch**

```python
# Setup the device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

```python
# Build the TinyVGG architecture
class TinyVGG(nn.Module):
    def __init__(self, in_channels=3, hidden_units=10, output_shape=10):
        super().__init__()
        # First conv layer
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Halves the spatial dimensions (128 -> 64)
        )
        
        # Second conv layer
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Halves the spatial dimensions again (64 -> 32)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 32 * 32, out_features=hidden_units)
        )
        
    def forward(self, X: torch.Tensor):
        X = self.conv_layer_1(X)
        X = self.conv_layer_2(X)
        X = self.classifier(X)
        return X
```

```python
# Create an instance of TinyVGG
torch.manual_seed(42)

model = TinyVGG(in_channels=3, hidden_units=10, output_shape=len(code)).to(device)
model
```

<center><img src="https://miro.medium.com/v2/resize:fit:1400/1*3ZkXJ-nIajuY3iX27w12aw.png" width="40%"></center>

```python
# Setup the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

# Setup the accuracy function using torchmetrics
accuracy_fn = Accuracy(task="multiclass", num_classes=len(code)).to(device)
```

```python
# Prepare dataloaders
BATCH_SIZE = 32
train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
pred_dataloader = DataLoader(TensorDataset(X_pred), batch_size=BATCH_SIZE, shuffle=False)

print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of pred_dataloader: {len(pred_dataloader)} batches of {BATCH_SIZE}")
```

```python
# Start the training loop
torch.manual_seed(42)

# Define the epochs
epochs = 50

# Store all the history in a dictionary
history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
}

for epoch in range(epochs):
    print(f"Epoch: {epoch} | ", end="")
    
    ## Training
    train_avg_loss, train_avg_acc = 0.0, 0.0
    
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        
        # Set the model in training mode
        model.train()
        
        # Perform the steps
        y_pred = model(X)                # Forward pass
        loss = loss_fn(y_pred, y)        # Calculate the loss
        train_avg_loss += loss.item()    # Accumulate loss as a scalar
        
        # Calculate accuracy
        acc = accuracy_fn(y_pred.argmax(dim=1), y).item()
        train_avg_acc += acc
        
        optimizer.zero_grad()            # Zero the gradients
        loss.backward()                  # Backpropagation
        optimizer.step()                 # Gradient descent
        
    # Divide total train_avg_loss and train_avg_acc by length of train dataloader
    train_avg_loss /= len(train_dataloader)
    train_avg_acc /= len(train_dataloader)
    
    ## Testing
    test_avg_loss, test_avg_acc = 0.0, 0.0
    model.eval()  # Set the model to evaluation mode
    
    with torch.inference_mode():  # Disable gradient computation for testing
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss = loss_fn(test_pred, y)
            test_avg_loss += test_loss.item()  # Accumulate test loss as a scalar
            
            # Calculate test accuracy
            test_acc = accuracy_fn(test_pred.argmax(dim=1), y).item()
            test_avg_acc += test_acc
        
    # Divide total test_avg_loss and test_avg_acc by length of test dataloader
    test_avg_loss /= len(test_dataloader)
    test_avg_acc /= len(test_dataloader)
    
    # Print out training and testing results
    history["epoch"].append(epoch)
    history["train_loss"].append(train_avg_loss)
    history["train_acc"].append(train_avg_acc)
    history["test_loss"].append(test_avg_loss)
    history["test_acc"].append(test_avg_acc)
    
    print(f"Train Loss: {train_avg_loss:.4f}, Train Accuracy: {train_avg_acc:.4f} | ", end="")
    print(f"Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_avg_acc:.4f}")
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes = axes.flatten()

sns.lineplot(x=history["epoch"], y=history["train_loss"], label="Train", ax=axes[0])
sns.lineplot(x=history["epoch"], y=history["test_loss"], label="Test", ax=axes[0])
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid();

sns.lineplot(x=history["epoch"], y=history["train_acc"], c="C2", label="Train", ax=axes[1])
sns.lineplot(x=history["epoch"], y=history["test_acc"], c="C3", label="Test", ax=axes[1])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid()

plt.tight_layout();
```

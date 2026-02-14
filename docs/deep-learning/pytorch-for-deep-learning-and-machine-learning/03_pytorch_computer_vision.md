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
    name: python3
---

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/geonextgis/PyTorch-for-Deep-Learning-and-Machine-Learning/blob/main/03_pytorch_computer_vision.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="zU1b4P1boxfE" -->
# **PyTorch Computer Vision**
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png" width="20%">
<!-- #endregion -->

<!-- #region id="kmarAk1Bo5K8" -->
## **0. Computer Vision Libraries in PyTorch**

* [`torchvision`](https://pytorch.org/vision/stable/index.html#torchvision) - base domain library in PyTorch computer vision
* `torchvision.datasets` - get datasets and data loading functions for computer vision here
* `torchvision.models` - get pretrained computer vision models that you can leverage for your own problems
* `torchvision.transforms` - functions for manipulating your vision data (images) to be suitable for use with an ML model
* `torch.utils.data.Dataset` - base dataset class for PyTorch
* `torch.utils.data.DataLoader` - creates a Python iterable over a dataset
<!-- #endregion -->

```python id="fHaIpITTNRJL" colab={"base_uri": "https://localhost:8080/"} outputId="600e2093-1b9c-439e-b5a8-56b5a32f95a8"
%pip install torchmetrics
```

```python colab={"base_uri": "https://localhost:8080/"} id="pTjoZNP5qkuf" outputId="c08a2f9f-1d42-4ce2-f204-c32fec9ef865"
import numpy as np
import pandas as pd
import random
from mlxtend.plotting import plot_confusion_matrix
from pathlib import Path

# Import PyTorch
import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from torchmetrics import ConfusionMatrix
from timeit import default_timer as timer
from tqdm.auto import tqdm

# Import matplotlib for visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Times New Roman"

# Check versions
print(torch.__version__)
print(torchvision.__version__)
```

<!-- #region id="ShP8Rsn5rXiV" -->
## **1. Getting a Dataset**
The dataset we'll be using is FashionMNIST from `torchvision.datasets` - https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hCoVFgUBrg-W" outputId="9c5d8550-56d5-41ee-eba0-5dbf6f79a66f"
# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # do we want the training dataset?
    download=True, # do we want to download yes/no?
    transform=transforms.ToTensor(), # how do we want to transform the data?
    target_transform=None # how do we want to transform the labels/targets?
    )

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
    target_transform=None
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="CbUJkiJ_6Yq-" outputId="c1548499-72ae-4f41-bf16-645f9340d21a"
len(train_data), len(test_data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="wD9t44dm6jsp" outputId="9e827681-0c5d-42d6-f7ea-2ac902336f2e"
# See the first training example
image, label = train_data[0]
image, label
```

```python colab={"base_uri": "https://localhost:8080/"} id="dNcmQrV27MVV" outputId="f36bd750-54fc-4cd6-d1b5-13e6affcb363"
class_names = train_data.classes
class_names
```

```python colab={"base_uri": "https://localhost:8080/"} id="MfJRraIF7U2e" outputId="5d1f1d11-ee8b-48f1-f578-8e6a32ca2a35"
class_to_idx = train_data.class_to_idx
class_to_idx
```

```python colab={"base_uri": "https://localhost:8080/"} id="rqSIorEg7ek-" outputId="e6a0aa84-6a2a-48f5-8886-b680b5b5e2c8"
train_data.targets
```

<!-- #region id="AyhnDgX97_RS" -->
### **1.1 Check the Input and Output Shape of the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="W6JCsMuc7h3e" outputId="df3d9991-5591-41eb-b439-3243de0c74e3"
# Check the shape of our image
print(f"Image shape: {image.shape} -> [color_channels, height, width]")
print(f"Image label: {class_names[label]}")
```

<!-- #region id="w8LtYvnU7z5B" -->
### **1.2 Visualizing Our Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 468} id="DIOiXtt08O2r" outputId="b5067966-a3d9-4a23-f177-6cefa5176234"
image, label = train_data[0]
print(f"Image shape: {image.shape}")
plt.imshow(image.squeeze())
plt.title(label);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 427} id="oFnLzf5J8psG" outputId="15fbb556-7e89-42f2-f83a-42896af9f76d"
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 751} id="RA3ek0l985zS" outputId="263c611a-cc47-4b3d-f4ab-5a82c86bb834"
# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False);
```

<!-- #region id="Kvl15q_g-w2z" -->
Do you think these items of clothing (images) could be modelled with pure linear lines? Or do you think we'll need non-linearities?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yXHRzDzH_DH4" outputId="28ce171f-b7f2-4587-aab3-564f50a4634a"
train_data, test_data
```

<!-- #region id="NYw97c8C-_7A" -->
## **2. Prepare DataLoader**

Right now, our data is in the form of PyTorch Datasets.

DataLoader turns our dataset into a Python iterable.

More specifically, we want to turn our data into batches (or mini-batches).

Why would we do this?

1. It is more computationally efficient, as in, your computing hardware may not be able to look (store in memory) at 60000 images in one hit. So we break it down to 32 images at a time (batch size of 32).

2. It gives our neural network more chances to update its gradients per epoch.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Vdt_Al_l_IUo" outputId="4ee4962d-4626-4500-ba99-69ae2a603967"
# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

train_dataloader, test_dataloader
```

```python colab={"base_uri": "https://localhost:8080/"} id="MAc4_y4jBYH4" outputId="fd06c2b0-f65d-4aeb-d5fc-45bc05fb5416"
# Let's check out what we've created
print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}...")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}...")
```

```python colab={"base_uri": "https://localhost:8080/"} id="ns8ovbbeCFmO" outputId="468e93d1-9d16-4f63-d510-21c2f5722d78"
# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 462} id="xRCKN3HVB-9d" outputId="9fb737e5-ab4b-40ec-899a-3ae80fd0cdea"
# Show a sample
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
print(f"Image Size: {img.shape}")
print(f"Label: {label}, Label Size: {label.shape}");
```

<!-- #region id="B82cO22eDpvp" -->
## **3. Model 0: Build a Baseline Model**

When starting to build a series of machine learning modelling experiments, it's best practice to start with a baseline model.

A baseline model is a simple model you will try to improve upon with subsequent models/experiments.

In other words: start simply and add complexity when necessary.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="j51m2jBAELHc" outputId="696a0951-b34b-4830-ab17-6261a3a190d7"
# Create a flatten layer
flatten_model = nn.Flatten()

# Get a single sample
X = train_features_batch[0]

# Flatten the sample
output = flatten_model(X) # perform forward pass

# Print out what happened
print(f"Shape before flattening: {X.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")
```

```python id="UqWo9KKDFDIy"
class FashionMNISTV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, X):
        return self.layer_stack(X)
```

```python colab={"base_uri": "https://localhost:8080/"} id="76rkQPT0GJlS" outputId="0e6469e0-3542-4b8d-985a-3f66bfa7e3d9"
torch.manual_seed(42)

# Setup model with input parameters
model_0 = FashionMNISTV0(
    input_shape=28*28, # this is 28*28
    hidden_units=10, # how many units in the hidden layer
    output_shape=len(class_names) # one for every class
).to("cpu")

model_0
```

```python colab={"base_uri": "https://localhost:8080/"} id="Pwq89ce5HjqU" outputId="830ae897-a36f-4bb8-dc9c-933b39800b63"
dummy_X = torch.rand([1, 1, 28, 28])
model_0(dummy_X)
```

```python colab={"base_uri": "https://localhost:8080/"} id="7gT2Zs_ZLu8I" outputId="24c93927-3d2b-4bb2-f2ea-a578f7c25668"
model_0.state_dict()
```

<!-- #region id="WTRVeWMbMOOF" -->
### **3.1 Setup Loss, Optimizer and Evauation Metrics**

* Loss function - since we're working with multi-class data, our loss function will be `nn.CrossEntropyLoss()`
* Optimizer - our optimizer `toch.optim.SGD()` (stochastic gradient descent)
* Evaluation metric - since we're working on a classification problem, let's use accuracy as our evaluation metric
<!-- #endregion -->

```python id="ntGsZBiUMaq_"
# Setup accuracy function using torchmetrics
accuracy_fn = Accuracy(task="multiclass", num_classes=len(class_names))

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
```

<!-- #region id="5qGDK50jOY9B" -->
### **3.2 Creating a Function to Time Our Experiments**

Machine learning is very experimental.

Two of the main things you'll often want to tracj are:
1. Model's performance (loss and accuracy values etc)
2. How fast it runs
<!-- #endregion -->

```python id="A-NSoTJTOq-1"
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time
```

```python colab={"base_uri": "https://localhost:8080/"} id="BOjZs9xZPiXL" outputId="359a680a-a009-4116-e1b5-1709cfa9c678"
start_time = timer()
# some code...
end_time = timer()
print_train_time(start=start_time, end=end_time, device="cpu")
```

<!-- #region id="PpNHJnDQQA5_" -->
### **3.3 Creating a Training Loop and Training a Model on Batches of Data**

1. Loop through epochs.
2. Loop through training batches, perform training steps, calculate the train loss per batch.
3. Loop through testing batches, perform testing steps, calcualte the loss per batch.
4. Print out what's happening.
5. Time it all (for fun).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 552, "referenced_widgets": ["46fb79d7ff8e46468a66763661f667f8", "fb3f54468b984eff9a07cc9ea336ce86", "92613ab9ec5942be99ff53a55ea51fa8", "368597cc83f64b54b427f3d96bb47ffb", "c1a3d6b82cfa49dfa5e9c3f91ddb7bf3", "f2b0a6b5a1c241c8bd66b31bf7b24ba2", "a1e9c66d43fc4170b1166ae96ad6f0dc", "45d0ce1ea2344e6780e049616ac219d7", "e2be8d90a71c4098938d9514e43312bb", "78cbed12396e4362ae9c1872eb18e3eb", "37226f2a127e4a2a9b47971f2c417039"]} id="WxmprMsZQqIa" outputId="0ffc84b0-bc17-4c4f-be27-fa2621d36ca2"
# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs (we'll keep this small for faster training time)
epochs = 3

# Create training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")

    ### Training
    train_loss = 0
    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()

        # 1. Forward pass
        y_pred = model_0(X)

        # 2. Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Perform backpropagation
        loss.backward()

        # 5. Perform gradient descent
        optimizer.step()

        # Print out what's happening
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

    # Divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader)

    ### Testing
    test_loss, test_acc = 0, 0
    model_0.eval()

    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X_test)

            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y_test)

            # 3. Calculate the accuracy
            test_acc += accuracy_fn(test_pred.argmax(dim=1), y_test).item()

        # Calculate the test loss average per batch
        test_loss /= len(test_dataloader)

        # Calculate the test acc average per batch
        test_acc /= len(test_dataloader)

    # Print out what's happening
    print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {(test_acc*100):.4f}%")

# Calculate training time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))
```

<!-- #region id="qyenQMtPWCGv" -->
## **4. Make Predictions and Get Model Results**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 101, "referenced_widgets": ["8e35ab42007b4f4f9a3464ca57f730b7", "0a014f2a3b2d4e4c8a6f58665a2d0656", "5b0557124d5a41f994f2eaf50254420c", "a0eca6f670c94c38b8e30f840c80f8c3", "8bb7c0c913da4124b7a8688c3a5b204a", "96537e09e8c94b9bab320ba5d07de221", "75e20440e5724e88be3fa154fd939a1f", "b27a01528eef4ceeba6369ca9f920982", "ef50dd6d42a14a9b895e3c34d7d226f1", "f783cd87f72840269bfa7f46ecb92003", "589d1f95af784094af0414da06047767"]} id="fv_3AYSUWJc5" outputId="6f7dc14e-0ca4-4606-c765-2e16a9518a9a"
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device):
    """Returns a dictionary containing the results of model predicting on data loader."""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Make our code device-agnostic
            X, y = X.to(device), y.to(device)
            # Make prediction
            y_pred = model(X)

            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_pred.argmax(dim=1), y)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class,
            "model_loss": loss.item(),
            "model_acc": acc.item()}

# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device="cpu")
model_0_results
```

<!-- #region id="0P4IGS1QzuBh" -->
## **5. Setup Device-Agnostic Code (for using a GPU if there is one)**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 36} id="VT5i3zytz6gT" outputId="f2588d2f-390c-4130-9889-21e60083e20c"
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

```python colab={"base_uri": "https://localhost:8080/"} id="u5p2om29z3g0" outputId="5e53528c-72e2-428b-a5b9-062ee41f833d"
!nvidia-smi
```

<!-- #region id="rQ0QEj730FtT" -->
## **6. Model 1: Building a Better Model with Non-linearity**
<!-- #endregion -->

```python id="5bYqE1Qb0lsD"
# Create a model with non-linear and linear layers
class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # flatten inputs into a single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, X: torch.Tensor):
        return self.layer_stack(X)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Dq9KK-hj2VZd" outputId="cab83a6a-c22d-467d-d5e1-c237eaff3e92"
# Create an instance of model_1
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784, # this is the output of the flatten after our 28*28 image goes in
                              hidden_units=10,
                              output_shape=len(class_names)).to(device) # send to the GPU if it's available

next(model_1.parameters()).device
```

<!-- #region id="fvY-lJO93LMj" -->
### **6.1 Setup Loss, Optimizer, and Evaluation Metrics**
<!-- #endregion -->

```python id="0RvBNi_d3SM1"
# Setup the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss() # measure how wrong our model is
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1) # tries to update our model's parameter to reduce the loss

# Setup the accuracy function
accuracy_fn = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)
```

<!-- #region id="JaSvsicR4C9S" -->
### **6.2 Functionizing Training and Evaluation/Testing Loops**
Let's create a function for:
* training loop - `train_step()`
* testing loop - `test_step()`
<!-- #endregion -->

```python id="4UUJeg3R4cak"
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """Performs a training with model trying to learn on data loader."""

    ### Training
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(data_loader):
        # Put the data on target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate the loss and accuracy (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        acc = accuracy_fn(y_pred.argmax(dim=1), # go from logits -> prediction labels
                          y).item()
        train_acc += acc

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Perform backpropagation
        loss.backward()

        # 5. Perform gradient descent (update the model's parameter once *per batch*)
        optimizer.step()

    # Divide total train loss and acc by length of train dataloader
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    # Print out what's happening
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")
```

```python id="8eg8ZGUTCJ_q"
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):

    """Performs a testing loop step on model going over data_loader"""

    test_loss, test_acc = 0, 0

    # Put the model in eval mode
    model.eval()

    #  Turn on inference model context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send the data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass (output raw logits)
            test_pred = model(X)

            # 2. Calculate the loss/acc
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(test_pred.argmax(dim=1), y).item() # go from logits -> prediction labels

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["1c8f6270216848ec9145e2ebd4da258e", "707103e56fc9416bbe85c71514e49b44", "b410aedd46764c9881b9c6307169cd4a", "55b0531aa186458699d0f29912d5568b", "de1185f1e5ab4093aae8d5cb4868e7ed", "36bb51bcba534b8abd0c02874d3a4a8d", "c7ac0a9fb01545829b164b8ea1ba0f63", "d8d66eefc8724408baa8540affa0a948", "9e8e563ade054a32969f93ed3a61d62f", "bfec5ef55da3423285cf6de35ab8757f", "9b834ccd7d2d46be8198156361413c79"]} id="Bv7vH5nxYSGM" outputId="3b394f0c-bcaf-42bd-f070-ea73068f31b2"
torch.manual_seed(42)

# Measure time
train_time_start_on_gpu = timer()

# Set epochs
epochs = 3

# Create an optimization and evaluation loop using train_step() and test_step()
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)
```

<!-- #region id="Gh9xIcENbV2V" -->
**Note**: Sometimes, depending on your data/hardware you might find that your model trains faster on CPU than GPU.

Why is this?
1. It copuld be that the overhead for copying data/model to and from the CPU outweighs the compute benefits offered by the GPU.
2. The hardware you're using has a better CPU in terms compute capability than the GPU.
3. For more on how to make your models compute faster, see here: https://horace.io/brrr_intro.html
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="n6uRO_EOb734" outputId="536f2412-3289-4e13-d01b-7ee4ef005f0a"
model_0_results
```

```python colab={"base_uri": "https://localhost:8080/", "height": 101, "referenced_widgets": ["6491d02bc72d453cacc41e694631d60a", "138fd122fe36440998444261a33dc2a4", "ea2613791cc44e0b9ffc77f2038bb158", "e523d985fe09442aaf92cab08ef0aaf0", "763f97d20e8d47f2811063c41650de7f", "7ac4157b12324542af41399365270b57", "8b329f5c6e08420d95b771f75c865906", "e1ed827543a04419b53d8062dbdedeac", "8dd86df2f96c4b5aafc12c932b1b6c03", "c50b2f333648465e8829b04bd30cd3f8", "fc92225652464039abff56f17e8e9152"]} id="tv4Wt1S7cfx6" outputId="22369434-1091-4800-f819-64880e8a62b6"
# Get model_1 result dictionary
model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device=device)
model_1_results
```

<!-- #region id="3vkkYa7ndYxp" -->
## **7. Model 2: Building a Convolutional Neural Network (CNN)**
CNN's are also known ConvNets.

CNN's are known for their capabilities to find patterns in visual data.

To find out what's happening inside a CNN, see this website: https://poloclub.github.io/cnn-explainer/
<!-- #endregion -->

```python id="PC6Qx5Pmhqto"
# Create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture that replicates the TinyVGG
    model from CNN explainer website
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1), # values we can set ourselves in our NN's are called hyperparameters
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, # there's a trick to calculating this.
                      out_features=output_shape)
        )

    def forward(self, X):
        X = self.conv_block_1(X)
        # print(f"Output shape of conv_block_1: {X.shape}")
        X = self.conv_block_2(X)
        # print(f"Output shape of conv_block_2: {X.shape}")
        X = self.classifier(X)
        # print(f"Output shape of classifier: {X.shape}")
        return X
```

```python colab={"base_uri": "https://localhost:8080/"} id="2ct9e1RElVh4" outputId="eda3d745-854b-4303-f883-1f904adf7925"
image.shape
```

```python id="DSeWRDY_iAeF"
torch.manual_seed(42)

model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="OXhpuhuR6oMc" outputId="c8808efd-aece-4189-def7-fa0832fa9081"
plt.imshow(image.squeeze(), cmap="gray");
```

```python colab={"base_uri": "https://localhost:8080/"} id="BpK6OOpq67b-" outputId="ba09ea01-86cb-421e-aebc-2208b6a17e2c"
rand_image_tensor = torch.randn(size=(1, 28, 28))
rand_image_tensor.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="G4weVyG77FZH" outputId="42b994c3-af9f-40b8-c4d4-08bbb5dcde7e"
# Pass image through model
model_2(rand_image_tensor.unsqueeze(dim=1).to(device))
```

```python id="c8M3lzxczqnv"
model_2.state_dict()
```

<!-- #region id="EyvHGY8vyaSQ" -->
### **7.1 Stepping through `nn.Conv2d()`**

See the documentation for nn.Conv2d() here - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FjABAIB3zIIb" outputId="d4e88367-9b97-43d1-9556-9ca934d6d718"
torch.manual_seed(42)

# Create a batch of images
images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]

print(f"Image batch shape: {images.shape}")
print(f"Single image shape: {test_image.shape}")
print(f"Test image:\n {test_image}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="3BF2HvQM0EVR" outputId="ae3108d9-2f88-4b0e-c9af-8fbaf48e6961"
test_image.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="BxCNS36x0IHy" outputId="7998ae41-0173-478f-d843-e2052d1daf73"
torch.manual_seed(42)

# Create a single conv2d layer
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=(3, 3),
                       stride=1,
                       padding=1)

# Pass the data through the convolutional layer
conv_output = conv_layer(test_image)
conv_output.shape
```

<!-- #region id="HJw1ZVgZ2GNz" -->
### **7.2 Stepping through `nn.MaxPool2d()`**
See the documentation for `nn.MaxPool2d()` here - https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="reMCtC3f2iTZ" outputId="956601ea-2369-42d2-9ca9-71a405debdb6"
test_image.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="Cb6vuATq2lfm" outputId="a0466944-c966-4094-8d2a-cb022a729a6c"
# Print out original image shape
print(f"Test image original shape: {test_image.shape}")

# Create a sample nn.MaxPool2d layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just the conv_layer
test_image_through_conv = conv_layer(test_image)
print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")

# Pass the data through max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="41tVqBcO4qvv" outputId="ab9b0a94-b2c5-499f-8046-3a615898372a"
torch.manual_seed(42)

# Create a random tensor with a similar number of dimension to our images
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"\nRandom tensor:\n{random_tensor}")
print(f"Random tensor shape: {random_tensor.shape}")

# Create a max pool layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass the random tensor through the max pool layer
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMax pool tensor:\n {max_pool_tensor}")
print(f"Max pool tensor shape: {max_pool_tensor.shape}")
random_tensor
```

<!-- #region id="Ver8MRzE-HpL" -->
### **7.3 Setup a Loss Function and Optimizer for `model_2`**
<!-- #endregion -->

```python id="Z7f88Zq5-R5E"
# Setup loss function/eval_metrics/optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)
accuracy_fn = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)
```

<!-- #region id="hWgYMeX4-233" -->
### **7.4 Training and Testing `model_2` using our Training and Test Functions**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["3f1a35f51a5c45c9afd8ba9b4d644aea", "bc6c2812690a430e820ca95afd3cbd03", "711a0e22bd7b4cdea856a3d25c6c1c35", "a0a145406acd4b60b1a1d385d9765cc4", "7b3e3f28b1eb47e2b11fc284cf09c721", "0368c8ebd5cf4a599550e64fecfeb643", "258904cd5f91469babe6332808b74265", "18b3a6ea380349df8549c281c1c83486", "a56412a4b5514a83845512b5f325164e", "3586c44fd8054fc39906d6ab74a46d24", "29ba4913a10447949d6978fb07fecbd2"]} id="t2CZLCuT_UGD" outputId="c2bc4ab4-a5a5-4291-a8de-168a2035587f"
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Measure time
from timeit import default_timer as time
train_time_start_model_2 = timer()

# Train and test model
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(model=model_2,
               data_loader=train_dataloader,
               loss=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)

    test_step(model=model_2,
              data_loader=test_dataloader,
              loss=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                            end=train_time_end_model_2,
                                            device=device)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 101, "referenced_widgets": ["a307aba396d74abea8d1902b342fc1fd", "d308807dbf7942c18e7507e89925c0e7", "ece2cf07559141e290b92f6a59cecb75", "3c855d88b24541ae8c07a34d8dad69f0", "32666e0c42fe48c08e4e87db3a8b1c6e", "8f44016a44ae4059b37dba8a0622fb06", "e12d6d8f4dd94713b1dc438c3131f0fb", "acfbc38ab24849ec8f8e13cf913b0857", "6502f00dd0494eceb3507b30a9388169", "cb444794ad204cb696055a694b7d30c7", "7384214fa5e34883bdd8965d66a5587a"]} id="ulHwmEy9B_OP" outputId="11f5df05-7309-41da-d8c0-eb9beaa2743c"
# Get model_2 results
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device
)

model_2_results
```

```python colab={"base_uri": "https://localhost:8080/"} id="xB-KrzAICz0M" outputId="a346e1ca-2037-4ec3-f8a4-fd37b4609f4b"
model_0_results
```

<!-- #region id="aqNQQaMUC29x" -->
## **8. Compare the Model Results and Training Time**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 143} id="7_4rUmrlC_6a" outputId="dc32ad31-a39b-449a-ec7f-f438b25ade6d"
compare_results = pd.DataFrame([model_0_results,
                                model_1_results,
                                model_2_results])

compare_results
```

```python colab={"base_uri": "https://localhost:8080/", "height": 143} id="R5YdbGI3Dls2" outputId="df6f3117-50fc-4748-f9a2-d7810b8b1263"
# Add training time to results comparison
compare_results["training_time"] = [total_train_time_model_0,
                                    total_train_time_model_1,
                                    total_train_time_model_2]
compare_results["model_acc"] = compare_results["model_acc"] * 100
compare_results
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="Q17OPhbfLY79" outputId="f7a5421d-f3e3-4922-86de-cc747f83028a"
# Visualize our model results
sns.barplot(data=compare_results, x="model_acc", y="model_name", width=0.5)
plt.xlabel("Accuracy (%)")
plt.ylabel("Model");
```

<!-- #region id="P-ItYRmAMSbZ" -->
## **9. Make and Evaluate Random Predictions with Best Model**
<!-- #endregion -->

```python id="tS0DXzzmMYnI"
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare the sample (add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass (model outputs raw logits)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred_prob off the GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6kXCP8A2NipI" outputId="99487a1b-47ef-44d2-f827-5f9ab19efcd2"
# random.seed(42)

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# View the first sample shape
test_samples[0].shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 451} id="2qMIMGGDO00_" outputId="f26a800d-9e3a-47fc-d12f-236b56b86c75"
plt.imshow(test_samples[0].squeeze(), cmap="gray")
plt.title(class_names[test_labels[0]]);
```

```python colab={"base_uri": "https://localhost:8080/"} id="POagx165PDi5" outputId="2492a0cf-304e-4491-93e4-679c8f0bb532"
# Make predictions
pred_probs = make_predictions(model=model_2,
                               data=test_samples)

# View first two prediction probabilities
pred_probs[:2]
```

```python colab={"base_uri": "https://localhost:8080/"} id="D4kUThLfPXar" outputId="9c69a142-efd6-43ba-d33c-5399ad0bbdac"
# Convert prediction probabilities to labels
pred_classes = pred_probs.argmax(dim=1)
pred_classes
```

```python colab={"base_uri": "https://localhost:8080/"} id="XhWID5SNPhWU" outputId="7a7bd6c9-f998-4ba0-bc6b-cd000c298729"
test_labels
```

```python colab={"base_uri": "https://localhost:8080/", "height": 749} id="fY3wjfrwPrls" outputId="bee9ad67-2028-489e-ce50-ef87348b39e4"
# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # Create subplot
    plt.subplot(nrows, ncols, i+1)

    # Plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")

    # Find the prediction (in text form, e.g "Sandal")
    pred_label = class_names[pred_classes[i]]

    # Get the truth label (in text form)
    truth_label = class_names[test_labels[i]]

    # Create a title for the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # Check for equality between pred and truth and change color of title text
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")
    else:
        plt.title(title_text, fontsize=10, c="r")

    plt.axis(False)
```

<!-- #region id="IJllHXI8RqBB" -->
## **10. Making a Confusion Matrix for Further Prediction Evaluation**

A confusion matrix is a fantastic way of evaluating your classification models visually:
1. Make predictions with our trained model on the test dataset
2. Make a confusion matrix `torchmetrics.ConfusionMatrix`
3. Plot the confusion matrix using `mlxtend.plotting.plot_confusion_matrix()`
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["022d24263bbc4d848afa125ccae25a5b", "acf78b19fd734a438a9fa80cfd0ba7f5", "e2e473f90eca47bc825b58248e96c499", "8aeb8dd41234409a890d1cc4f70b2dd4", "2345eb5376034efda4319b939d2c6a9e", "cd28069fbb68491db96bd83841de16ad", "895ae70cd7044019a4e80b4b6b74ecb2", "3afbff32079f4e1f82b570b402b9d10e", "2331b8d427c943bb81bccd97b824f74a", "236bd6d190134df2950b1abfbe3d5098", "6d0e7fcdad1b45de9a7bc63e5cca6071"]} id="AcjDJmOwShvy" outputId="a909d7fc-3285-4f16-d596-5167a441d3da"
# Make predictions with trained model
y_preds = []
model_2.eval()

with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions..."):
        # Send the data andtargets to target device
        X, y = X.to(device), y.to(device)
        # Forward pass
        y_logit = model_2(X)
        # Turn predictions from logits -> prediction probabilities -> prediction labels
        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
        # Put prediction on CPU for evaluation
        y_preds.append(y_pred.cpu())

# Concatenate list of predictions into a tensor
# print(y_preds)
y_pred_tensor = torch.cat(y_preds)
y_pred_tensor
```

```python colab={"base_uri": "https://localhost:8080/"} id="2EGOAPV2T0J1" outputId="d87774ee-e2c1-4344-8c96-48e58a33b3a4"
len(y_pred_tensor)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 668} id="NZNW5NbST2pc" outputId="218a9acb-2500-4b49-91af-838ab11ed52d"
# Setup confusion instance and compare prediction to targets
confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with numpy
    class_names=class_names,
    figsize=(10, 7)
)
```

<!-- #region id="Pjbg3ol-WQIE" -->
## **11. Save and Load Best Performing Model**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mZXS8PdoWZwM" outputId="5bf4da1a-2603-41a3-e418-db7afcbd4796"
# Create model directory path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# Create model save
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(),
           f=MODEL_SAVE_PATH)
```

```python colab={"base_uri": "https://localhost:8080/"} id="N18TPPs4Xs7c" outputId="a67dbb25-cf33-4028-dc45-490fcfa34a34"
# Create a new instance
torch.manual_seed(42)

loaded_model_2 = FashionMNISTModelV2(input_shape=1,
                                     hidden_units=10,
                                     output_shape=len(class_names))

# Load in the save_dict()
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send the model to the target device
loaded_model_2.to(device)
```

```python colab={"base_uri": "https://localhost:8080/"} id="OvZ9lN51Ym6L" outputId="4b152d41-6be8-423d-fb9e-444c0a2981df"
model_2_results
```

```python colab={"base_uri": "https://localhost:8080/", "height": 101, "referenced_widgets": ["dcb21b8949de46cbb8990bf1dd65237e", "277d758935cc45f3a0e9810963bb3a6e", "d4de2fe3bbb54c929a0581b7063620fa", "026fa826a2dc46b382b1857f173505ad", "6d286f91810047a4b886bbfd1abc8750", "b510869e4e484648b46f79563debd832", "8c1801dba96b412dac4de19defff672f", "d899ff8da4064d65b00e210f2aad8080", "9409cc7fb0b345008ccb60101ef7feaf", "ba630d6c7cf54ceeba0c3bfd3da23f76", "11e8b22a81694c2498822ca4684f46f9"]} id="HxMCX6fUYpID" outputId="a16d1a4b-5e03-46a1-ff13-0e25650efc76"
# Evaluate the loaded model
torch.manual_seed(42)

loaded_model_2_results = eval_model(
    model=loaded_model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device
)

loaded_model_2_results
```

```python colab={"base_uri": "https://localhost:8080/"} id="WloZtkejZLjt" outputId="abe1e8ec-5377-402b-c127-594bd26daa2c"
# Check if model results are close to each other
torch.isclose(torch.tensor(model_2_results["model_loss"]),
              torch.tensor(loaded_model_2_results["model_loss"]),
              atol=1e-02)
```

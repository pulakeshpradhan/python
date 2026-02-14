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

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/geonextgis/PyTorch-for-Deep-Learning-and-Machine-Learning/blob/main/02_pytorch_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="wHF-br6nc3Kq" -->
# **Neural Network Classification with PyTorch**
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png" width="20%">

Classification is a problem of predicting whether something is one thing or another (there can be multiple things as the outputs).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 36} id="cAUzVso-dz0_" outputId="7c4acda5-938b-4a28-f578-81c068b97444"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import requests
from pathlib import Path

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = "Times New Roman"

import warnings
warnings.filterwarnings("ignore")

torch.__version__
```

<!-- #region id="z1Xfvor2dC2r" -->
## **1. Make Classification Data and Get Ready**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9_e-EL06eUyr" outputId="567fba31-0331-487f-f08b-fcd7df8e2e40"
# Make 1000 samples
n_samples =1000

# Create circles
X, y = make_circles(n_samples=n_samples,
                    noise=0.03,
                    random_state=42)

len(X), len(y)
```

```python colab={"base_uri": "https://localhost:8080/"} id="PsgpYT4BexaG" outputId="8d00e215-da19-46ad-8393-0510f455c093"
print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 223} id="OQ0fMsQjfBZP" outputId="a35f94bf-36ae-4239-93c7-482c786c4695"
# Make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
print(circles.shape)
circles.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="h1h7t7UKfvkl" outputId="7a83f37f-a9df-44ee-9fa4-ff534ccf79e5"
# Visualize, visualize, visualize
sns.scatterplot(data=circles, x="X1", y="X2", hue="label");
```

<!-- #region id="2CyE1XD7g57V" -->
**Note:** The data we're working is often referred to as a toy dataset, a dataset that is small enough to experiment but still sizeable enough to practice the fundamentals.
<!-- #endregion -->

<!-- #region id="vHNI0O2mhLzv" -->
### **1.1 Check Input and Output Shapes**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="e708iA00hUUp" outputId="546b0a5a-3f10-4910-9022-8d7745739cd7"
X.shape, y.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="NX2Alu8BhXWA" outputId="d3ca74f8-e0f7-4b45-e512-1475a1a85706"
# View the first sample example features and labels
X_sample = X[0]
y_sample = y[0]

print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")
```

<!-- #region id="sZ0B5xzwh552" -->
### **1.2 Turn Data into Tensors and Create Train and Test Splits**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HLtdvlJ9iCeN" outputId="57a09dca-2ebc-4373-b203-f39eaa8a7d15"
# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X[:5], y[:5]
```

```python colab={"base_uri": "https://localhost:8080/"} id="UXeCPLJUidr_" outputId="8a7e370d-c875-4bef-9cc8-121db057ad53"
type(X), X.dtype, y.dtype
```

```python colab={"base_uri": "https://localhost:8080/"} id="MJND7cB1iou4" outputId="87a1771d-52b4-4f04-a139-f02e99c9939e"
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

len(X_train), len(X_test), len(y_train), len(y_test)
```

<!-- #region id="GD6TtMosjbDu" -->
## **2. Building a Model**

Let's build a model to classify our blue and orange dots.

To do so, we want to:
1. Setup device agnostic code so our code will run on an accelerator (GPU) if there is one
2. Construct a model (by subclassing `nn.Module`)
3. Define a loss function and optimizer
4. Create a training and test loop
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 36} id="-HsXsKZNj7B7" outputId="f20f09b6-70cd-4785-9e29-f455c1adb4d4"
# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

<!-- #region id="NcYbokQfkQU9" -->
Now we've setup device agnostic code, let's create a model that:
1. Subclasses `nn.Module` (almost all models in PyTorch subclass `nn.Module`)
2. Create 2 `nn.Linear()` layers that are capable of handling that shapes of our data.
3. Defines a `foward()` method that outlines the forward pass (or forward computation) of the model
4. Instantiate an instance of our model class and send it to the target `device`
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="AIcAMaVzlMfD" outputId="3562e5ad-4b27-4de1-b865-e8bf138a5573"
# 1. Construct a model that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handing the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features and upscales to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features from previous layer and outputs a single features (same shape as y)

        # self.two_linear_layers = nn.Sequential(
        #     nn.Linear(in_features=2, out_features=5),
        #     nn.Linear(in_features=5, out_features=1)
        # )

    # 3. Define a forward() method that outlines the forward pass
    def forward(self, X):
        return self.layer_2(self.layer_1(X)) # X -> layer_1 -> layer_2 -> output
        # return self.two_linear_layers(X)

# 4. Instantiate an instance of our model class and send it to the target device
model_0 = CircleModelV0().to(device)
model_0
```

```python colab={"base_uri": "https://localhost:8080/", "height": 36} id="uAFz8UVmnG5W" outputId="9d0c91dc-6bfc-494b-86ef-c02ecf53ae3d"
device
```

```python colab={"base_uri": "https://localhost:8080/"} id="t0q3up4JoVwB" outputId="74f1dfab-4e61-441e-fd61-3d4c26375e2e"
next(model_0.parameters()).device
```

```python colab={"base_uri": "https://localhost:8080/"} id="OvUGh9NlocHy" outputId="ec2ee29b-c5ea-4e9d-99a5-bc0c71efdb0e"
# Let's replicate the model above using nn.Sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

model_0
```

```python colab={"base_uri": "https://localhost:8080/"} id="X03vbWwJplWC" outputId="be4aba39-7169-42a9-8351-0b9d533fc8b0"
model_0.state_dict()
```

```python colab={"base_uri": "https://localhost:8080/"} id="cFy0-wp-p55D" outputId="856c58ca-cc91-48fb-db6c-5e03d1a7dadf"
# Make predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
print(f"\nFirst 10 predictions: \n{torch.round(untrained_preds[:10])}")
print(f"\nFirst 10 labels:\n{y_test[:10]}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="4rwJc-c_q06s" outputId="3fd8dd92-4695-44ba-dfb7-6b2bae14bedc"
X_test[:10], y_test[:10]
```

<!-- #region id="7NsQf7YwrRMu" -->
### **2.1 Setup Loss Functions and Optimizer**
Which loss function or optimizer should you use?

Again... this is problem specific.

For example for regression you might want MAE or MSE (Mean Absolute Error or Mean Squared Error).

For classification you might want binary cross entropy or categorical cross entropy (cross entropy).

As a reminder, the loss function measures how wrong your models predictions are.

And for optimizers, two of the most common and useful are SGD and Adam, however PyTorch has many built-in optins.

* For the loss function we're going to use `totch.nn.BCEWithLogitLoss()`.
* For different optimizers see `torch.optim`
<!-- #endregion -->

```python id="yVqmpVaxrXOu"
# Setup the loss function
# loss_fn = nn.BCELoss() # BCELoss = requires input to have gone through the sigmoid  activation function prior to input to BCELoss
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitloss = sigmoid activation function built-in

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)
```

```python id="OFYy-JrWvgdl"
# Calculate accuracy - out of 100 examples, what percentage does our model get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
```

<!-- #region id="3a854i7wv9HQ" -->
## **3. Train Model**
To train our model, we're going to need to build a training loop:

1. Forward pass
2. Calculate the loss
3. Optimizer zero grad
4. Loss backward (backpropagation)
5. Optimizer step (gradient descent)
<!-- #endregion -->

<!-- #region id="xEu6QJD6wMs6" -->
### **3.1 Going from raw logits -> prediction probabilities -> prediction labels**

Our model outputs are going to be raw **logits**.

We can convert these **logits** into prediction probabilities by passing them to some kind of activation function (e.g. sigmoid for binary classification and softmax for multiclass classification).

Then we can convert our model's prediction probabilities to **prediction labels** by either rounding them or taking the `argmax()`.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CZUxJoWfwwN9" outputId="d630f153-af15-4247-97d2-0e98003dbf53"
# View the first 5 outputs of the forward pass on the test data
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]

y_logits
```

```python colab={"base_uri": "https://localhost:8080/"} id="BuvIH87Xx6wd" outputId="62826887-5964-4d5d-ec23-88eba045685d"
y_test[:5]
```

```python colab={"base_uri": "https://localhost:8080/"} id="W-WLTSJax9qN" outputId="1a056582-7075-4730-fdcc-d47f89804730"
# Use the sigmoid activation function on our model logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
y_pred_probs
```

<!-- #region id="S8MfLO16yWBx" -->
For our prediction probability values, we need to perform a range-style rounding on them:
* `y_pred_probs` >= 0.5, `y=1` (class 1)
* `y_pred_probs` < 0.5, `y=0` (class 0)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WROvVHmByPA3" outputId="9a8fd874-418b-4e3a-ac39-d247ec4aca3c"
# Find the predicted labels
y_preds = torch.round(y_pred_probs)

# In full (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# Get rid of extra dimension
y_preds.squeeze()
```

```python colab={"base_uri": "https://localhost:8080/"} id="EuJB4U1ezq_h" outputId="5ba4823f-65b2-4ea7-bc53-4a0846044157"
y_test[:5]
```

<!-- #region id="ssuG4DFcnv2Q" -->
### **3.2 Building a Training and Testing Loop**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Lj2dxp1Hn6_R" outputId="9de4ecbf-4c52-4124-d306-b44c4c66d633"
# Set the manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the number of epochs
epochs = 100

# Put the data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labels

    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # nn.BCELoss expects prediction probabilities as input
    #                y_train)
    loss = loss_fn(y_logits, # nn.BCEWithLogitLoss expects raw logits as input
                   y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward (backpropagation)
    loss.backward()

    # 5. Optimizer step (gradient descent)
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate test loss/acc
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

        # Print out what's happenin
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
```

<!-- #region id="URTtjCXYofBe" -->
## **4. Make Predictions and Evaluate the Model**
From the metrics it looks like our model isn't learning anything...

So to inspect it let's make some predictions and make them visual!

In othre words, "Visualize, visualize, visualize!"

To do so, we're going to import a function called `plot_decision_boundary()`
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wKTvOIZbpah6" outputId="2682afec-8169-41b3-a862-584ab793d6c1"
# Download helper functions from Learn PyTorch repo (if it's not already downloaded)
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary
```

```python colab={"base_uri": "https://localhost:8080/", "height": 544} id="jt5h1qfitBWL" outputId="5e0509f5-7b9d-43fa-f000-f2ea5caa4e44"
# Plot the decision boundary of the model
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
```

<!-- #region id="keQi06wpt6Ii" -->
## **5. Improving a Model (from a Model Perspective)**

* Add more layers - give the model more chances to learn about patterns in the data
* Add more hidden units - go from 5 hidden units to 10 hidden units
* Fit for longer
* Changing the activation functions
* Change the learning rate
* Change the loss function

These options are all from a model's perspective because they deal directly with the model, rather than the data.

And beacuse these options are all values we (as machine learning engineers and data scientists) can change, they are referred as **hyperparameters**.

Let's try and improve our model by:
* Adding more hidden units: 5 -> 10
* Increase the number of layers: 2 -> 3
* Increase the number of epochs: 100 -> 1000
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="UcG-bMtvvqjO" outputId="17551b92-bd4c-48db-9023-b50b75048000"
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, X):
        # z = self.layer_1(X)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(X))) # this way of writing operations leverages speed ups where possible behind the scene

model_1 = CircleModelV1().to(device)
model_1
```

```python id="eTVkJVXmxwi-"
# Create a loss function
loss_fn = nn.BCEWithLogitsLoss()

# Create an optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="2um5kRcUyGBA" outputId="c6a3dabe-3e46-403b-b631-4163c6ff4a66"
# write a training and evaluation loop for model_1
epochs = 1000

# Put data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    ### Training
    model_1.train() # Set the model in training model

    # 1. Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2. Calculate the loss
    loss = loss_fn(y_logits,
                   y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation
    loss.backward()

    # 5. Perform gradient descent
    optimizer.step()

    ### Testing
    with torch.inference_mode():
        model_1.eval() # Set the model the the eval mode

        # Forward pass
        test_logits = model_1(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        # Calculate test loss/acc
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)

        # Print out what's happening
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 544} id="P6rpQtiO3JXT" outputId="4109177a-8c26-4af6-ec10-24f3f1750f35"
# Plot the decision boundary of the model
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
```

<!-- #region id="44DdAkxy32XI" -->
### **5.1 Preparing Data to See if Our Model can Fit a Straight Line**

One way to troubleshoot to a larger problem is to test out a smaller problem.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xHjua31I39PY" outputId="db265a3c-d8b8-4ca8-bd36-f00ffa3ca77e"
# Create some data
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.01

# Create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias # Linear regression formula (without epsilon)

# Check the data
print(len(X_regression))
X_regression[:5], y_regression[:5]
```

```python colab={"base_uri": "https://localhost:8080/"} id="IuiR77zr47nv" outputId="c0cd7a81-14e0-4133-b277-01449994f238"
# Create train and test splits
train_split = int(len(X_regression) * 0.8)
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# Check the lengths of each
len(X_train_regression), len(y_train_regression), len(X_test_regression), len(y_test_regression)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 599} id="MsxlF8Ki56rn" outputId="9b9127fd-aa31-4482-b297-ad0ee1f448ea"
# Plot the data
plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression)
```

<!-- #region id="sbcj-c_76ejZ" -->
### **5.2 Adjusting `model_1` to Fit a Straight Line**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nOQUDPq466-l" outputId="e113d9e9-20b9-417e-878b-9b85ae944048"
# Same architecture as model_1 (but using nn.Sequential)
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

model_2
```

```python id="pIbPEuHw7ZB5"
# Loss and Optimizer
loss_fn = nn.L1Loss() # MAE loss with regression data
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.01)
```

```python colab={"base_uri": "https://localhost:8080/"} id="_2r4dHD97n7J" outputId="cc67b360-1cb1-4ce8-e349-a69d98833148"
# Train the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the number of epochs
epochs = 1000

# Put the data on the target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

### Training
for epoch in range(epochs):
    model_2.train()

    # 1. Forward pass
    y_pred = model_2(X_train_regression)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train_regression)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation
    loss.backward()

    # 5. Perform gradient descent
    optimizer.step()

    ### Testing
    model_2.eval()

    with torch.inference_mode():
        test_preds = model_2(X_test_regression)

    test_loss = loss_fn(test_preds, y_test_regression)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 599} id="LF9-_iG5-0VY" outputId="79d24c6f-2d3d-4f17-9789-e7f1671dada0"
# Turn on evaluation mode
model_2.eval()

# Make predictions (inference)
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

# Plot data and predictions
plot_predictions(train_data=X_train_regression.cpu(),
                 train_labels=y_train_regression.cpu(),
                 test_data=X_test_regression.cpu(),
                 test_labels=y_test_regression.cpu(),
                 predictions=y_preds.cpu())
```

<!-- #region id="R-IC5X96idrS" -->
## **6. The Missing Piece: Non-Linearity**

"What patterns could you draw if you were given an infinite amount of a straight and non-straight lines?"

Or in machine learning terms, an infinite (but really it is finite) or linear and non-linear functions?
<!-- #endregion -->

<!-- #region id="XloUXWUAxjMF" -->
### **6.1 Recreating Non-Linear Data (Orange and Blue Circles)**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="r48X73nfxsp2" outputId="d43184ea-c1ed-462d-a1f3-ebae53e04bad"
# Make and plot data
n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y);
```

```python colab={"base_uri": "https://localhost:8080/"} id="uN9sDBZzyKZB" outputId="2432a0ae-3766-42e9-e9e6-4b652b01f702"
# Convert data to tensors and then to train and test splits
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

X_train[:5], y_train[:5]
```

<!-- #region id="YHNwEng7y8It" -->
### **6.2 Building a Model with Non-Linearity**
* Linear = Straight Line
* Non-Linear = Non-Straight Lines

Artifical neural networks are a large combination of linear (straight) amd non-straight (non-linear) functions which are potentiallly able to find patterns in the data.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="X19rKkr8zdO7" outputId="04623d65-3b7e-4403-ecfa-a03920f76a6a"
# Build a model with non-linear activation function
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # relu is a non-linear activation function

    def forward(self, X):
        # Where should we put our non-linear activation functions?
        return self.layer_3(self.relu(self.layer_2(self.layer_1(X))))

model_3 = CircleModelV2().to(device)
model_3
```

```python id="p4xe0Hj80yTl"
# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(), lr=0.1)
```

<!-- #region id="uAr7yTSI2PtP" -->
### **6.3 Train a Model with Non-Linearity**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="YX0kIvep2W-4" outputId="219e5d00-4351-47c2-dcab-b676b4d366c9"
# Random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Put all the data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Loop through the data
epochs = 1000

for epoch in range(epochs):
    ### Training
    model_3.train()

    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2. Calculate the loss
    loss = loss_fn(y_logits,
                   y_train)

    acc = accuracy_fn(y_train, y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation
    loss.backward()

    # 5. Perform gradient descent
    optimizer.step()

    ### Testing
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy_fn(y_test, test_preds)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
```

<!-- #region id="BPfam_JI6vgB" -->
### **6.4 Evaluating a Model Trained with Non-Linear Activation Functions**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Y-TW5j196-Jo" outputId="6dadce69-d778-4435-d168-1c52126c42c1"
# Make predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test).squeeze()))

y_preds[:10], y_test[:10]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 544} id="_F-qSSnm7Ttz" outputId="1a2768b3-877f-480e-98e3-7e4b0e53a01f"
# Plot the decision boundaries
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)
```

<!-- #region id="1t1cy0Tn8BO8" -->
## **7. Relicating Non-Linear Activation Functions**

Neural networks, rather than us telling the model what to learn, we give it the tools to discover patterns in the data and it tries to figure out the patterns on its own.

And these tools are linear & non-linear functions.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="g1nyY6bL8Gxp" outputId="68b7ca5d-0093-4c06-9337-506ed9ee0388"
# Create a tensor
A = torch.arange(-10, 10, 1, dtype=torch.float32)
A.dtype
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="3vBuE4pH8nCT" outputId="3d6a91b0-cfef-44fa-8651-8495a9bb2a60"
# Visualize the tensor
plt.plot(A);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="MwrBBp9f8rgE" outputId="edcb96de-1346-4f2e-80fb-762298a1abed"
plt.plot(torch.relu(A));
```

```python colab={"base_uri": "https://localhost:8080/"} id="ahgMxlBM9CLu" outputId="e48ae914-48b1-4169-9061-b6c3fa89d1df"
def relu(X: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0), X) # input must be a tensor

relu(A)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="EYoz6X6S9SKo" outputId="d4c6a5ca-1734-4375-b8cb-480af8d01de6"
# Plot ReLU activation function
plt.plot(relu(A));
```

```python id="N5OgiYVK9a1q"
# Now let's do the same for sigmoid
def sigmoid(X: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-X))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="V1jb1n6I9vkj" outputId="f9dc3f85-5d31-48fb-b395-c854fa8ee027"
plt.plot(torch.sigmoid(A));
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="_CkrTjs2901M" outputId="c03e700a-441d-4f4a-9e49-0918dc89f783"
plt.plot(sigmoid(A));
```

<!-- #region id="viUuD269p5bE" -->
## **8. Putting it all Together with a Multi-Class Classification Problem**
* Binary Classification = One thing or another (cat vs. dog, spam vs. not spam, fraud or not fraud)
* Multi-class classification = More than one thing or another (cat vs. dog vs. chicken)
<!-- #endregion -->

<!-- #region id="HXuUfQEfqahA" -->
### **8.1 Creating a Toy Multi-Class Dataset**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 36} id="gvtgci1Kq1kh" outputId="4d12c320-21a9-4b01-f8e4-263b5026cb36"
# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
from torch import nn

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = "Times New Roman"

torch.__version__
```

```python colab={"base_uri": "https://localhost:8080/", "height": 465} id="beyq0_bgrm7W" outputId="1093ecc7-809f-405f-fb10-03ed56cdccea"
# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5, # give the clusters a little shake up
                            random_state=RANDOM_SEED)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# 3. Split the data into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# 4. Plot data (visualize, visualize, visualize)
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_blob[:, 0], y=X_blob[:, 1], hue=y_blob, palette="Set1", edgecolor="k");
```

<!-- #region id="7FEq0oSat5_-" -->
### **8.2 Building a Multi-Class Classification Model in PyTorch**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 36} id="2PupUWRwrlov" outputId="b0a7f2ae-77f2-48b5-9195-ec9e2f371988"
# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

```python colab={"base_uri": "https://localhost:8080/"} id="o6JHL6C4ugw2" outputId="df4ea65d-0f6a-42bd-f4e5-35c6525e1597"
# Build a multi-class classification model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes multi-class classification model.

        Args:
            input_features (int): Number of input features to the model
            output_features (int): Number of output features (number of output classes)
            hidden units (int): Number of hidden units between layers, default 8

        Returns:

        Example:
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, X):
        return self.linear_layer_stack(X)

# Create an instance of BlobModel and send it to the target device
model_4 = BlobModel(input_features=2,
                    output_features=4,
                    hidden_units=8).to(device)

model_4
```

<!-- #region id="xnWJK4fKw6NG" -->
### **8.3 Create a Loss Function and Optimizer for a Multi-Class Classification Model**
<!-- #endregion -->

```python id="sG8UD6rGxlo5"
# Create a loss function for multi-class classification
loss_fn = nn.CrossEntropyLoss()

# Create an optimizer for multi-class classification
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1) # learning rate is a hyperparameters you can change
```

<!-- #region id="PCh0fj2MyA5U" -->
### **8.4 Getting Prediction Probabilities for a Multi-Class Pytorch Model**

In order to evaluate and train and test our model, we need to convert our model's output (logits) to prediction probabilties and then to predcition labels.

Logits (raw output of the model) -> Pred probs (use torch.softmax) -> Pred labels (take the argmax of the prediction probabilities)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xxwcG8DJyPZV" outputId="e696b272-3f59-44df-c650-5aeb1e3ed621"
# Put the data on the target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

# Let's get some raw outputs of our model (logits)
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

y_logits[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="tgUVg1A5y_BT" outputId="9acbf7fa-c65d-42da-900c-947c354d25c1"
y_blob_test[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="lPYLHTHBzdiP" outputId="040e1643-6d57-497f-f912-ab54b253dd40"
# Convert our model's logit outputs to prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
print(y_pred_probs[:5])
```

```python colab={"base_uri": "https://localhost:8080/"} id="sEReT3ie0CJz" outputId="ddc7f695-a574-4ab6-cc18-d6b895385d14"
# Convert our model's prediction probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)
y_preds
```

```python colab={"base_uri": "https://localhost:8080/"} id="x-5AbB8K0GzP" outputId="37993381-8203-4d04-f552-95c3c63db98c"
y_blob_test
```

<!-- #region id="mgexSV7r0oOF" -->
### **8.5 Creating a Training Loop and Testing Loop for a Multi-Class Classification**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tCOgCv1Z1ENR" outputId="ba8368c8-f9a9-49ef-f178-5db33931ba2d"
# Fit the multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

for epoch in range(epochs):
    ### Training
    model_4.train()

    # 1. Forward pass
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # 2. Calculate the loss
    loss = loss_fn(y_logits,
                   y_blob_train)

    acc = accuracy_fn(y_blob_train,
                      y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation
    loss.backward()

    # 5. Perform gradient descent
    optimizer.step()

    ### Testing
    model_4.eval()

    # Forward pass
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        # Calculate the loss and accuracy
        test_loss = loss_fn(test_logits,
                            y_blob_test)

        test_acc = accuracy_fn(y_blob_test, test_preds)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss:.4f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
```

<!-- #region id="NPTJHfR15q2x" -->
### **8.6 Making and Evaluating Predictions with a PyTorch Multi-Class Model**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="p1COdl8B5zpr" outputId="f54e6f7c-d2c6-4746-8dd4-81c5181b56e9"
# Make predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1) # Logits -> Probabilities -> Class

# View the first 10 predictions
y_preds[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="nYahYVDy6ODl" outputId="8f768696-3a7c-4dd0-e015-7485b6b1072c"
y_blob_test[:10]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 544} id="Ns85ZKjo5a_x" outputId="7282b87f-1253-4608-c739-687782b2ec51"
# Plot the decision boundary
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
```

<!-- #region id="Kcn_qsrV7VXn" -->
## **9. A Few More Classification Metrics... (to Evaluate our Classification Model)**

* Accuracy: out of 100 samples, how many does our model get right?
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report

If you want access to a lot of PyTorch metrics, see TorchMetrics - https://lightning.ai/docs/torchmetrics/stable/
<!-- #endregion -->

```python id="lQ6UfsrH850g"
!pip install torchmetrics
```

```python colab={"base_uri": "https://localhost:8080/"} id="dFa4m03Y81NB" outputId="bdc5884f-820a-4188-e2f7-75c15cecc799"
from torchmetrics import Accuracy

# Setup metric
torchmetric_accuracy = Accuracy(task="multiclass", num_classes=4).to(device)

# Calculate accuracy
torchmetric_accuracy(y_preds, y_blob_test)
```

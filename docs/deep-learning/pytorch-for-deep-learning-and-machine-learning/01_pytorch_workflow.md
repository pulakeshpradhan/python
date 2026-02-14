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
<a href="https://colab.research.google.com/github/geonextgis/PyTorch-for-Deep-Learning-and-Machine-Learning/blob/main/01_pytorch_workflow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="tJxN_prNayY5" -->
# **PyTorch Workflow**
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png" width="20%">

A typical workflow for a machine learning project using PyTorch involves several key steps, from data preparation to model deployment. Below is an outline of a common PyTorch workflow:

1. **Get Data Ready (Turn into Tensors):**<br>
    The first step involves preparing your dataset. This includes loading your data and transforming it into a format that PyTorch can work with, specifically tensors. Tensors are multidimensional arrays that are the basic building blocks in PyTorch, allowing for efficient computation on GPUs.

2. **Build or Pick a Pretrained Model (to Suit Your Problem)**<br>
    In this step, you either build a custom model from scratch or select a pretrained model that fits the task at hand. Pretrained models can be especially useful when working with large, complex datasets like images or text. This step also involves:

    - **Pick a Loss Function & Optimizer**: Selecting an appropriate loss function that the model will try to minimize and choosing an optimizer that will update the model parameters during training.
    - **Build a Training Loop**: Setting up a loop that will iterate over the data in batches, feed it through the model, compute the loss, and adjust the model's parameters to minimize the loss.

3. **Fit the Model to the Data and Make a Prediction**<br>
    In this phase, the model is trained on the prepared data. The training loop defined earlier is executed, allowing the model to learn from the data by minimizing the loss function. Once the model has been trained, it can make predictions on new, unseen data.

4. **Evaluate the Model**<br>
    After training, the model's performance is assessed on a validation or test dataset. This step determines how well the model has learned and whether it generalizes well to new data. The evaluation results help identify any issues, such as overfitting or underfitting.

5. **Improve Through Experimentation**<br>
    Based on the evaluation results, the model may need to be improved. This could involve experimenting with different model architectures, hyperparameters, or data preprocessing techniques. The goal is to fine-tune the model for better performance.

6. **Save and Reload Your Trained Model**<br>
   Once the model has been trained and evaluated successfully, it is saved to disk. This allows you to reload the model later for further training, fine-tuning, or deployment. Saving the model also ensures that you don't need to retrain it every time you want to make predictions.

This workflow is iterative, meaning that based on the results from the evaluation and improvement steps, you may need to loop back and refine earlier stages.

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01_a_pytorch_workflow.png">

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 36} id="WLBC4RYrZXUl" outputId="0fe53de5-ad96-4450-c831-f351611923be"
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "deJavu Serif"
plt.rcParams["font.serif"] = "Times New Roman"

torch.__version__
```

<!-- #region id="XbbrFOCzZ-05" -->
## **1. Data Preparation and Loading**
Data can be almost anything... in machine learning.
- Excel spreadsheet
- Images of any kind
- Videos (YouTube has lots of data)
- Audio like songs or podcasts
- DNA
- Text

Machine learning is a game of two parts:
1. Get data into a numerical representation.
2. Build a model to learn patterns in that numerical representation.

To showcase this, let's create some *known* data using the linear regression formula.

We'll use a linear regression formula to make a straight line with known **parameters**.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="r6_NuArnaUUj" outputId="70c2a1a2-101a-4c9c-cdfa-45a012e88179"
# Create *known* parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="okOgdLC2cfcS" outputId="94e56b6f-5455-465c-cd30-bf7cb6b25019"
len(X), len(y)
```

<!-- #region id="t52dbu_tchz6" -->
## **Splitting Data into Training and Test Sets**

(One of the most important concept in machine learning.)

Let's create a training and test set with our data.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jyAJ_BADheCt" outputId="84ddabf2-2e40-42b2-9d99-120eb1f40719"
# Create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)
```

<!-- #region id="6FuimE7uiEg6" -->
How might we better visualize our data?

This is where the data explorer's motto comes in!

"Visualize, visualize, visualize!"
<!-- #endregion -->

```python id="lmR8SHtGiXfE"
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(8, 5))

    # Plot the training data in plue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot the testing data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # Are there predictions?
    if predictions is not None:
        # Plot the predictions if they exis
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

```

```python colab={"base_uri": "https://localhost:8080/", "height": 445} id="RxvTf6Voj24n" outputId="e0faedbf-2a19-484c-928a-7069efa394d2"
plot_predictions()
```

<!-- #region id="FYhaqHm7kHfY" -->
## **2. Build the Model**

Our first PyTorch model!

This is very exciting... let's do it!

What our model does:
- Start with random values (weight & bias)
- Look at the training data and adjust the random values to better represent (or get closer to) the ideal values (the weight and bias values we used to create the data)

How does it do so?

Through two mail algorithms:
1. Gradient descent
2. Backpropagation
<!-- #endregion -->

```python id="ZJ7KRSUPkqv8"
# Create linear regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, # <- start with a random weight and try to adjust it to the ideal weight
                                                requires_grad=True, # <- can this parameter be updated via gradient descent?
                                                dtype=torch.float)) # <- PyTorch loves the datatype torch.float32

        self.bias = nn.Parameter(torch.randn(1, # <- start with a random weight and try to adjust it to the ideal BIAS
                                             requires_grad=True, # <- can this parameter be updated via gradient descent?
                                             dtype=torch.float)) # <- PyTorch loves the datatype torch.float32

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data
        return self.weights * x + self.bias # <- this is the linear regression formula
```

<!-- #region id="kCqJhiU1kqGb" -->
### **PyTorch Model Building Essentials**
- `torch.nn`: contains all of the buildings for computational graphs (a neural network can be considered as a computational graph)
- `torch.nn.Parameter`: what parameters should our model try and learn, often a PyTorch layer from `torch.nn` will set these for us
- `torch.nn.Modules`: The base class for all neural network modules, if you subclass it, you should overwrite forward()
- `torch.optim`: this is where the optimizers in PyTorch liv, they will help with gradient descent
- `def forward()`: All `nn.Module` subclasses require you to overwrite `forward()`, this method defines what happens in the forward computation.

See more of these essential modules via the PyTorch cheatsheet - https://pytorch.org/tutorials/beginner/ptcheat.html
<!-- #endregion -->

<!-- #region id="9Ea_WH5jGEy2" -->
### **Checking the Contents of a PyTorch Model**
Now we've created a model, let's see what's inside...

So we can check our model parameters or what's inside our model using `.parameters()`.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GaqzXBZmIjwW" outputId="e7646bab-aa4e-44d1-b831-01cbd592d899"
# Create a random seed
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module
model_0 = LinearRegressionModel()

# Check out the parameters
list(model_0.parameters())
```

```python colab={"base_uri": "https://localhost:8080/"} id="wAXJazW7Jb09" outputId="ed911d6c-c8e0-4850-e81e-675cb9bddcaf"
# List named parameters
model_0.state_dict()
```

```python colab={"base_uri": "https://localhost:8080/"} id="qniuLJvpJn4W" outputId="e3b6ff01-30b4-429c-b795-313bd2a9f2a3"
weight, bias
```

<!-- #region id="fWchWlNfJv7P" -->
### **Making Prediction using `torch.inference_mode()`**
To check our model's predictive power, let's see how well it predicts `y_test` based on `x_test`.

When we pass data through our model, it's going to run it through the `forward()` method.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="s181nXuTLSlz" outputId="bbc19761-d5bf-4a67-e7b9-c36e626970b3"
y_preds = model_0(X_test)
y_preds
```

```python colab={"base_uri": "https://localhost:8080/"} id="RWKhKsuJKhuN" outputId="656a01ea-ea6e-49fb-e07e-34be3a162463"
# Make predictions with model
with torch.inference_mode():
    y_preds = model_0(X_test)

# You can also do something similar with torch.no_grad(), however, torch.inference_mode() is preferred
with torch.no_grad():
    y_preds = model_0(X_test)

y_preds
```

```python colab={"base_uri": "https://localhost:8080/"} id="NNE7_FKvLDH5" outputId="839475a5-d286-41ae-bc7a-75e700ee51f0"
y_test
```

```python colab={"base_uri": "https://localhost:8080/", "height": 445} id="qZwUUhJ9LFJ5" outputId="ed00b61f-7177-4fda-afac-af9b3a5f3268"
plot_predictions(predictions=y_preds)
```

<!-- #region id="NlkzTXKYEyFJ" -->
## **3. Train Model**

The whole idea of training is for a model to move from some *unknown* parameters (these may be random) to some *known* parameters.

Or in other words from a poor representation of the data to a better representation of the data.

One way to measure how poor or how wrong your models predictions are is to use a loss function.

* Note: Loss function may also be called cost function or criterion in different areas. For our case, we're going to refer to it as a loss function.

Things we need to train:
* **Loss function:** A function to measure how wrong your model's predictions are to the ideal outputs, lower is better.
* **Optimizer:** Takes into account the loss of a model and adjusts the model's parameters (e.g., weight and bias) to improve the loss function.

    * Inside the optimizer you'll often have to set two parameters:
        - `params` - the model parameters you'd like to optimize, for example params=model_0.parameters()

        - `lr (learning rate)` - the learning rate is a hyperparameter that define how big/small the optimizer changes the parameters with each step (a small `lr` results in small changes, a larger `lr` results in large changes)

And specifcally for PyTorch, we need:
* A training loop
* A testing loop
<!-- #endregion -->

```python id="Yenl_ZBGE2p5" colab={"base_uri": "https://localhost:8080/"} outputId="8217390b-90ab-499f-ef68-4756a2d13a5c"
list(model_0.parameters())
```

```python colab={"base_uri": "https://localhost:8080/"} id="KxtOV01D0gIE" outputId="77a7215e-191a-44ec-ae5f-b439cac332a8"
# Check out our model's parameters (a parameter is a value that the model sets itself)
model_0.state_dict()
```

```python id="UEbZcQUi0yfc"
# Setup a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr=0.01) # lr = learning rate = possibly the most important hyperparameter you can set
```

<!-- #region id="w0VsXCerT1w2" -->
**Q:** Which loss function and optimizer should I use?

**A:** This will be problem specific. But with experience, you'll get an idea of what works and what doesn't with your particular problem set.

For example, for a regression problem (like ours), a loss function of `nn.L1loss()` and an optimizer like `torch.optim.SGD()` will suffice.

But for a classification problem like classifying whether a photo is of a dog or a cat, you'll likely want to use a loss function of `nn.BCELoss()` (binary cross entropy loss).
<!-- #endregion -->

<!-- #region id="O2y8w1ugUwl1" -->
### **Building a Training Loop (and a Testing Loop) in PyTorch**

A couple of things we need in a training loop:
0. Loop through the data
1. Forward pass (this involves data moving through our model's `forward()` functions) to make predictions on data - also called forward propagation.
2. Calculate the loss (compare forward pass predictions to ground truth labels)
3. Optimizer zero grad
4. Loss backward - move backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss (**backpropagation**)
5. Optimizer step - use the optimizer to adjust our model's parameters to try and improve the loss (**gradient descent**)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="uDPqjngEWc7G" outputId="5a1fbe5f-eac6-431d-b12c-9f3f5c536c16"
list(model_0.parameters())
```

```python colab={"base_uri": "https://localhost:8080/"} id="zICzYw3UU3RM" outputId="a688c341-b8bf-4b2a-f79c-483555d83e8e"
torch.manual_seed(0)

# An epoch is one loop through the data
epochs = 100

# Track different values
epoch_count = []
loss_values = []
test_loss_values = []

### Training
# 0. Loop through the data
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train() # train mode in PyTorch sets all parameters that require gradients to require gradients

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_train, y_pred)
    print(f"Loss: {loss}")

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step() # by default how the optimizer changes will accumulate through the loop so... we have to zero them above in step 3 for the next iteration of the loop

    ### Testing
    model_0.eval() # turns off different settings in the model not needed for evaluation/testing (dropout/batch norm layers)
    with torch.inference_mode():
        # 1. Do the forward pass
        test_pred = model_0(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    # Print out what's happening
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)

        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        # Print out model state_dict()
        print(model_0.state_dict())
```

```python colab={"base_uri": "https://localhost:8080/"} id="vjIoD7MdiveJ" outputId="b335cded-b353-4edb-829a-3792e93e8ba5"
epoch_count
```

```python colab={"base_uri": "https://localhost:8080/", "height": 452} id="cb2yNx_-h_Qb" outputId="c615a2e1-b456-4eb1-e0c0-25a5d529fff4"
# Plot the loss curves
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
plt.plot(epoch_count, np.array(test_loss_values), label="Test loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend();
```

```python id="SM51R97UdEwR"
with torch.inference_mode():
    y_preds_new = model_0(X_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 445} id="rNz82diMdSgQ" outputId="c02141c5-23da-4cb1-810c-25dbab6e1a82"
plot_predictions(predictions=y_preds_new)
```

<!-- #region id="i99vpGP1Ndbz" -->
## **4. Saving a Model in Pytorch**
There are three main methods you should about for saving and loading models in PyTorch.
1. `torch.save()` - allows you save a PyTorch object in Python's pickle format
2. `torch.load()` - allows you load a saved PyTorch object
3. `torch.nn.Module.load_state_dict()` - this allows to load a model's saved state dictionary
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="arwBMkkhNi_J" outputId="31d32aeb-dc0d-4be3-f4fb-7aef5f0dc132"
# Saving our PyTorch model
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)
```

```python colab={"base_uri": "https://localhost:8080/"} id="rcC9KGwbQGqS" outputId="afaa2a17-6d66-4b46-a727-68f9df16aee3"
!ls -l models
```

<!-- #region id="tT5cJfAeQL5s" -->
## **5. Loading a PyTorch Model**
Since we saved our model's `state_dict()` rather than entire model, we'll create a new instance of our model class and load the saved `state_dict()` into that.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Khq9reUgQtgY" outputId="6c0792ca-0f29-4b0f-e786-0738e7821297"
# To load in a saved saved state_dict we have to instantiate a new instance of our model class
loaded_model_0 = LinearRegressionModel()

# Load the saved state_dict of model_0 (this will update the new instance with updated parameters)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
```

```python colab={"base_uri": "https://localhost:8080/"} id="VWPrPAKeRmI-" outputId="8d26783c-0684-4c32-dc72-d49707919322"
# Make some preditions with our loaded model
loaded_model_0.eval()
with torch.no_grad():
    loaded_model_preds = loaded_model_0(X_test)

loaded_model_preds
```

```python colab={"base_uri": "https://localhost:8080/"} id="LuFUFO5nR66Z" outputId="d8a8e206-e4af-4834-b2e6-6d7000714079"
# Compare the loaded model preds with original model preds
y_preds == loaded_model_preds
```

```python colab={"base_uri": "https://localhost:8080/"} id="_u5-jN_mSHCS" outputId="e09cbe00-321f-4361-fe89-826ec27ea531"
# Make some models preds
model_0.eval()

with torch.inference_mode():
    y_preds = model_0(X_test)

y_preds
```

```python colab={"base_uri": "https://localhost:8080/"} id="8o1e_kOoStA2" outputId="284d7244-bfe5-49bb-bbe1-fd601ea841fb"
# Compare the loaded model preds with original model preds
y_preds == loaded_model_preds
```

<!-- #region id="PcxVq3nLXpCK" -->
## **6. Putting it all together**
Let's go back through the steps above and see it all in one place.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 36} id="OEUr-0tHX1za" outputId="2a28866f-1ac7-4b98-f595-57f213366697"
# Import PyTorch and matplotlib
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Check the PyTorch version
torch.__version__
```

<!-- #region id="lK_GWffqYXGO" -->
Create device-agnostic code.

This means if we've got access to a GPU, our code will use it (for potentially faster computing).

If no GPU is available, the code will default to using CPU.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="C9-6T5IQYyhB" outputId="d4586118-a2eb-42ff-ec87-69c98516cd84"
# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="5NXbg2_3ZSDE" outputId="058aebd1-7207-411d-a2d7-dbb50428f1ba"
!nvidia-smi
```

<!-- #region id="TwQ5kfhlZqFF" -->
### **6.1 Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FjuvKOoVZvPW" outputId="0675974d-ad90-4740-bd5d-c40d21fce6b6"
# Create some data using the linear regression formula of y = weight * X + bias
weight = 0.7
bias = 0.3

# Create X and y (features and labels)
X = torch.arange(start=0, end=1, step=0.02).unsqueeze(dim=1)
y = weight * X + bias

X[: 10], y[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="-YTZphXBaTrD" outputId="d9323799-8089-4eb4-8921-4dbf9e45ec65"
# Split the data
train_split = int(len(X) * 0.8)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 445} id="_C8yGfWQaxyf" outputId="92349562-fa68-4d2c-ad0e-4ebf035a5f09"
# Plot the data
# Note: if you don't have the plot_pedictions() function loaded, this will error
plot_predictions(X_train, y_train, X_test, y_test)
```

<!-- #region id="bOS3wKCIbGdq" -->
### **6.2 Building a PyTorch Linear Model**
<!-- #endregion -->

```python id="AKoEkF1YbaME" colab={"base_uri": "https://localhost:8080/"} outputId="ece76aa6-ce01-464b-ea72-07949cb06f81"
# Create a linear model by subclassing nn.Module
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()

        # Use nn.Linear() for creating the model parameters / also called: linear transform, probing layer, fully connected layer, dense layer
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(X)

# Set the manual seed
torch.manual_seed(42)
model_1 = LinearRegression()
model_1, model_1.state_dict()
```

```python colab={"base_uri": "https://localhost:8080/"} id="-ty1bOA_msT7" outputId="366452e5-3209-4180-de32-615c1cbaf86a"
# Check the model current device
next(model_1.parameters()).device
```

```python colab={"base_uri": "https://localhost:8080/"} id="QBCA6tBRnaCf" outputId="335d28f8-6749-46da-e8ee-042fb180df00"
# Set the model to use the target device
model_1.to(device)
next(model_1.parameters()).device
```

<!-- #region id="LMjUyylMnmgR" -->
### **6.3 Training**
For training we need:
* Loss function
* Optimizer
* Training Loop
* Testing Loop
<!-- #endregion -->

```python id="Ri2MnVZCn2Ir"
# Setup the loss function
loss_fn = nn.L1Loss()

# Setup our optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)
```

```python colab={"base_uri": "https://localhost:8080/"} id="-DTU_Dl1oRQt" outputId="cb5b7b56-f5c4-465a-e85b-09fab5f5fef5"
# Let's write a training loop
torch.manual_seed(42)

epochs = 200

# Put data on the target device (device agnostic code for data)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    # Set the model in the training mode
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_train, y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation
    loss.backward()

    # 5. Perform gradient descent
    optimizer.step()

    ### Testing
    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="-pJdlbZXrisE" outputId="844166aa-a06c-4f9f-8b80-69c554322719"
model_1.state_dict()
```

```python colab={"base_uri": "https://localhost:8080/"} id="zsnOtj5krnHU" outputId="d6aa7027-c16d-4f88-c9ac-ea856b5a483b"
weight, bias
```

<!-- #region id="eR2iUrb3rpNt" -->
### **6.4 Making and Evaluating Predictions**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WxwBoIpmsGrg" outputId="d65daf7e-5db5-4856-fd68-135148a7f8da"
# Turn model into evaluation mode
model_1.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_1(X_test)

y_preds
```

```python colab={"base_uri": "https://localhost:8080/", "height": 445} id="lOw-c0VHsdbT" outputId="cfcac21f-e60f-473d-c5a9-1f2cde8ee11e"
plot_predictions(predictions=y_preds.cpu())
```

<!-- #region id="Dkrvr_8GsuLM" -->
### **6.5 Saving and Loading a Trained Model**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="A2xXgctTs1qN" outputId="225bb41e-f11b-4cc3-9b36-49a60842d6b8"
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)
```

```python colab={"base_uri": "https://localhost:8080/"} id="zAmF0iaotyut" outputId="af64f4cf-355d-4b85-968e-846954062547"
# Load a PyTorch model

# Create a new instance of linear regression model
loaded_model_1 = LinearRegression()

# Load the saved model_1 state dict
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))

# Put the loaded model to device
loaded_model_1.to(device)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ed55DLAruXrB" outputId="7694abfc-c1e6-4c98-95ac-8c033c85dc8e"
next(loaded_model_1.parameters()).device
```

```python colab={"base_uri": "https://localhost:8080/"} id="9n9mO1fFur5k" outputId="4060c6cc-1fd9-46f5-a659-4533b4a250e5"
loaded_model_1.state_dict()
```

```python colab={"base_uri": "https://localhost:8080/"} id="j8LPm5DmuuPy" outputId="f91fe6c4-b222-4bf9-9438-9348221a70ae"
# Evaluate the loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)

y_preds == loaded_model_1_preds
```

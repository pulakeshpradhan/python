[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/deep-learning/pytorch-for-deep-learning-and-machine-learning/projects/00_simple_linear_regression_using_pytorch.ipynb)

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

<!-- #region id="nonSJo40jzM7" -->
# **Simple Linear Regression using PyTorch**
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png" width="20%">
<!-- #endregion -->

<!-- #region id="gC9sAr3lj99x" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="AEiwvpPWtWDT" outputId="77848429-147b-4b82-caba-4faf860d042e"
from google.colab import drive, userdata
drive.mount("/content/drive")
```

```python colab={"base_uri": "https://localhost:8080/"} id="xe_ZdxCYuRYH" outputId="64fa4da6-cf29-4b58-c5e3-50649d24021b"
# Download the data from kaggle
!kaggle datasets download -d saquib7hussain/experience-salary-dataset
```

```python colab={"base_uri": "https://localhost:8080/"} id="XLHH7b5EvePQ" outputId="9efc0c5b-1852-42d0-825e-69605de94b45"
# Unzip the dataset
!unzip /content/experience-salary-dataset.zip
```

```python id="uO38bMnzrGsY"
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = "Times New Roman"

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="xQktDQkTu260" -->
## **Read the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 223} id="WsXoQ1WIvngB" outputId="9f27fc4e-ddd1-457d-87a4-6597bfffd6e3"
# Read the data
data = pd.read_csv("/content/Experience-Salary.csv")
print(data.shape)
data.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 147} id="97RDQODkvzu7" outputId="98a05d2f-d8b0-4f40-8c22-decee3979fa0"
# Check the NaN values
data.isnull().sum()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="Gy7mCfgdxTHV" outputId="df29976d-0c52-4124-e666-ff930e989875"
# Plot the data
sns.scatterplot(data=data, x="exp(in months)", y="salary(in thousands)");
```

<!-- #region id="DuDJO76Lv8Kc" -->
## **Train-Test-Split**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="o0-ZrL_yv_jM" outputId="1c91eb4e-02f9-43df-fbd0-8c65de410f38"
# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    data["exp(in months)"],
    data["salary(in thousands)"],
    test_size=0.3,
    random_state=42
)

X_train.shape, X_test.shape
```

```python id="9lsm9CsJ57Yj"
X_train, X_test = torch.tensor(X_train.to_numpy()), torch.tensor(X_test.to_numpy())
y_train, y_test = torch.tensor(y_train.to_numpy()), torch.tensor(y_test.to_numpy())
```

<!-- #region id="zYUw_3Oyxj5J" -->
## **Build the Model using PyTorch**
<!-- #endregion -->

```python id="AzBAdweJxreQ"
# Create a LinearRegression class
class LinearRegression(nn.Module):
    # Constructor
    def __init__(self):
        super().__init__()
        # paremeters (weight and bias)
        self.weight = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    # Forward method to define the computation in the model
    def forward(self, X):
        return self.weight * X + self.bias
```

```python colab={"base_uri": "https://localhost:8080/"} id="k7rvz4ip2BSO" outputId="e10befce-4332-4d94-cc1c-e9607fcdd9c7"
torch.manual_seed(42)

# Instantiate a model object
model = LinearRegression()

# Print the initial parameters
list(model.parameters())
```

```python id="LKjA7JRd1p_t"
# Define the loss function and the optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.001)
```

```python colab={"base_uri": "https://localhost:8080/"} id="8c69dGWl2dEL" outputId="cd023ace-a04e-4832-b31c-2ef4ece1efcc"
# Train the model
epochs = 100

# Store the epoch results
epoch_count = []
train_loss_values = []
test_loss_values = []

## Training loop
for epoch in range(epochs):

    # Set the model to training mode
    model.train()

    # 1. Forward pass
    y_pred = model(X_train)

    # 2. Calculate the training loss
    train_loss = loss_fn(y_train, y_pred)

    # 3. Zero the gradients
    optimizer.zero_grad()

    # 4. Backward pass
    train_loss.backward()

    # 5. Perform gradient descent (update parameters)
    optimizer.step()

    ## Testing
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        # 1. Forward pass on the test set
        test_pred = model(X_test)

        # 2. Calculate the test loss
        test_loss = loss_fn(y_test, test_pred)

    # Append epoch and loss values (detach to avoid computation graph)
    epoch_count.append(epoch)
    train_loss_values.append(train_loss.item())  # Use .item() to get the scalar value
    test_loss_values.append(test_loss.item())    # Use .item() to get the scalar value

    # Print out epoch number and loss values
    print(f"Epoch: {epoch+1} | Train Loss: {train_loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="kzooYiNn7dRb" outputId="dfddfb38-b9f0-4ffd-efea-59db11acf69b"
plt.plot(epoch_count, train_loss_values, label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show();
```

<!-- #region id="N_F1rxeTQ3Eu" -->
## **Plot the Regression Line**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="q8WtTzYmRbZ3" outputId="a6f62ab5-731b-4561-a75c-669f73ec7e8e"
with torch.no_grad():
    preds = model(data.iloc[:, 0].values)

# Plot the data
sns.scatterplot(data=data, x="exp(in months)", y="salary(in thousands)")
plt.plot(data.iloc[:, 0].values, preds, c="r", label="Regression line")
plt.legend()
plt.show();
```

<!-- #region id="yeVQ730BS4ur" -->
## **Save the Model**
<!-- #endregion -->

```python id="0M1NGEKoS7ii"
torch.save(model.state_dict(), "lr_model.pth")
```

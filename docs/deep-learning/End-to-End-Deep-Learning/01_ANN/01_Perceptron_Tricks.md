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

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/01_ANN/01_Perceptron_Tricks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="ca93464a-a7e1-4eb1-a2c8-cc31997723f0" -->
# **Perceptron Trick: How to Train a Perceptron**
<!-- #endregion -->

<!-- #region id="686e3f84-118e-4848-a236-a62ebfa9b66c" -->
## **Import Rquired Libraries**
<!-- #endregion -->

```python id="i9u_ADko9370"
# %pip install ipympl
```

```python id="efeb428f-2098-4830-a24e-eca2062a9f2f"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="a91dd2ce-1e0d-4350-b778-7871baf9bca9" -->
## **Create a Dataset for Classification**
<!-- #endregion -->

```python id="d1948dd7-0088-4d85-a296-8773c602a4dc"
# Create a classification dataset with two features
X, y = make_classification(n_samples=100,
                           n_features=2,
                           n_informative=1,
                           n_redundant=0,
                           n_classes=2,
                           n_clusters_per_class=1,
                           random_state=0,
                           hypercube=False,
                           class_sep=1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 452} id="6cd476fd-8c8a-42ee-badf-98e1065080e7" outputId="b7d28603-6c4b-465b-ca71-87489ed77fd9"
# Plot the data
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, s=50)
plt.title("Classification Dataset")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="388bcccc-8b34-4453-92c3-149f48e617fe" outputId="b24ab15c-eac0-424c-c619-44e75de4460f"
# Print the first 5 rows of x
X[:5, :]
```

```python colab={"base_uri": "https://localhost:8080/"} id="4dec12e1-4160-4376-a8df-662eaf135a55" outputId="ad335202-facd-4c82-f30a-61f579bae7f0"
# Print the first 5 items of y
y[:5]
```

<!-- #region id="70e71a6e-9f6e-4c64-b1db-9ad91c804e32" -->
## **Create a Function to Implement Perceptron Algorithm**
<!-- #endregion -->

<!-- #region id="2cddae4f-f7b5-43e9-a6a8-7cfd06c30c92" -->
#### **Formula to Update the Weights in a Peceptron Model:**
<center><img src="https://i.stack.imgur.com/qgovN.jpg" width="50%"></center>
<!-- #endregion -->

```python id="bca56c1a-55fc-4302-8334-b909179db1b9"
# Define a step function
def step(z):
    """
    This function returns 0 if value is less than or equals to 0 and returns 1
    if value is greater than 0.
    """
    return 0 if z <= 0 else 1
```

```python id="9e2eba2a-b135-4399-b807-978f7143186c"
def perceptron(X, y, epoch):
    # Add an extra column with the X to represent bias
    X = np.insert(arr=X, obj=0, values=1, axis=1)

    # Define a variable for weights
    weights = np.ones(X.shape[1])

    # Define the learning rate
    lr = 0.1

    for i in range(epoch):
        j = np.random.randint(0, 100)

        # Select a random row from x
        X_random = X[j]
        y_random = y[j]

        # Predict the y for x_random
        y_hat = step(np.dot(X_random, weights))

        # Update the weights
        weights = weights + lr*(y_random - y_hat)*X_random

    return weights
```

<!-- #region id="19a9abdb-90e6-48d6-b02e-a18179f73bba" -->
## **Apply the Perceptron Model on the Classification Dataset**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="a625909b-4d49-45a4-994a-b5ddf5f3e2e2" outputId="23062495-9db9-412e-a1ac-43d1a24db997"
# Apply the Perceptron model on the dataset and extract the weights
weights = perceptron(X, y, epoch=1000)
weights
```

```python colab={"base_uri": "https://localhost:8080/"} id="5619a711-80bb-42d1-839b-c012787e8224" outputId="c1ebb1fc-dbca-4ec0-eac6-5733cb8be297"
# Extract the intercept and coefficients value
intercept_ = weights[0]
coeff_ = weights[1:]
print("Intercept:", intercept_)
print("Coefficients:", coeff_)
```

<!-- #region id="1dd9c9d0-1230-4ce3-a4fc-f02bd8156f65" -->
## **Derive the Separation Line from the Coefficients and Intercept Value**
<!-- #endregion -->

<!-- #region id="4f5869c6-d661-479c-a00a-523b6d45e58a" -->
General Equation of a Line is:<br>
$$ \ Ax + By + C = 0 \ $$

We an also write it like:<br>
$$ \ y = mx + c \ $$ where $ \ m\ $ is the slope and $ \ c\ $ is the y-intercept.<br>
or,
$$ \ y = -\frac{A}{B}x - \frac{C}{B} \ $$
where $ -\frac{A}{B} \ $ is the slope and $ -\frac{C}{B} \ $ is the intercept.<br>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="36f49036-4977-477f-af6e-31ab7662090c" outputId="099f0b85-9990-450c-d641-abfe519201e5"
# Define the slope and intercept
m = -(coeff_[0]/coeff_[1])
print("Slope (m):", m)
```

```python colab={"base_uri": "https://localhost:8080/"} id="25ba112c-0e7c-4e80-9f37-79793d275606" outputId="c6f4d5d4-2261-47fd-8c3b-4b4df80ef8ef"
c = -(intercept_/coeff_[1])
print("Y-intercept (c):", c)
```

```python id="3eb1eecc-759c-44ec-9a84-58e08ba6761d"
# Plot the line on the scatter plot
X_input = np.linspace(start=-1, stop=1, num=100)
y_input = (m * X_input) + c
```

```python colab={"base_uri": "https://localhost:8080/", "height": 452} id="db9dc667-c4ac-42fe-95a7-72ef86e49b78" outputId="42fd545b-6adf-4cb5-bcc4-cfdb8e1e457c"
ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, s=50)
sns.lineplot(x=X_input, y=y_input, ax=ax, c="red")
plt.xlim((-1.1, 0.85))
plt.ylim((-3, 3))
plt.title("Perceptron Model")
plt.show()
```

<!-- #region id="3d83993c-5441-4465-b9da-dbbf65572ac1" -->
## **Visualize the Change in Model with an Animation**
<!-- #endregion -->

```python id="4f089f41-b6cd-46cd-b31f-b42387da089b"
def perceptron(X, y):
    m = []
    c = []

    X = np.insert(X, 0, 1, axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1

    for i in range(1000):
        j = np.random.randint(0, 100)
        y_hat = step(np.dot(X[j], weights))
        weights = weights + lr*(y[j] - y_hat)*X[j]

        m.append(-(weights[1]/weights[2]))
        c.append(-(weights[0]/weights[2]))

    return m, c
```

```python id="42cf285a-470a-450b-b821-028dee7f8edc"
m, c = perceptron(X, y)
```

```python id="01786bc5-a299-42cd-87b1-9d7b0e942ab3"
# Plot the animation
%matplotlib ipympl
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
```

```python colab={"base_uri": "https://localhost:8080/", "height": 589, "referenced_widgets": ["bfe3dd1a89b84fc29091f047cb952d66", "41604381b9c24837b2f5cb980a526a63", "4d937df3e58e44daaa0281a24f1b14cc", "a38174bfcac74ab1b80ea47331ee0fd5"]} id="a823da1b-c3a2-4ccf-a4ca-2ebc7be98d61" outputId="0740536a-bd18-4f32-ef00-24016ca69f4f"
fig, ax = plt.subplots(figsize=(9, 5))

x_i = np.arange(-1, 1, 0.1)
y_i = x_i*m[0] + c[0]
ax.scatter(X[:, 0], X[:, 1], s=50)
line, = ax.plot(x_i, x_i*m[0]+c[0], "r-", linewidth=2)
plt.ylim(-3, 3)
def update(i):
    label = f"epoch {i}"
    line.set_ydata(x_i*m[i] + c[i])
    ax.set_xlabel(label)

anim = FuncAnimation(fig, update, repeat=True, frames=1000, interval=1000000)
```

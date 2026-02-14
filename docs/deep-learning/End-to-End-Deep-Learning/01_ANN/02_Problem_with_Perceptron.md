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
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/01_ANN/02_Problem_with_Perceptron.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="76275fbc-103b-4a5d-b5df-fe5ba0c7f2e9" -->
# **Problem with Perceptron**
The perceptron is a basic type of artificial neural network used in machine learning and deep learning. It is a simple model that can be used for binary classification tasks.

**Limitations:**<br>

* **Limited to Linear Relationships:** Perceptrons can only model linear relationships between input features and the output. They cannot capture non-linear patterns in the data.

* **Inability to Solve Non-Linear Problems:** If your data is not linearly separable, a perceptron will not be able to learn and make accurate predictions.

* **Lack of Complexity:** Perceptrons cannot represent complex functions or understand intricate data patterns, making them unsuitable for tasks that require higher-level abstractions.

* **Poor for Image and Text Data:** In tasks involving images, text, or other high-dimensional data, perceptrons may struggle because these data types often contain non-linear structures.

* **Limited Feature Interaction:** Perceptrons treat features independently and do not capture interactions between them, which is a key limitation for many real-world problems.
<!-- #endregion -->

<!-- #region id="17cc993f-fef2-4266-ab04-97ad457c3ebb" -->
## **Import Required Libraries**
<!-- #endregion -->

```python id="c4c8a31b-8ab0-4c38-93f2-2d4b66e7cf6c"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="b3694b0a-ded1-4b86-b4fe-536f4bb12893" -->
## **Make the DataFrames**
<!-- #endregion -->

```python id="775254dc-9605-4607-9eb1-2a467f5f11ab"
or_dict = {"input1": [1, 1, 0, 0],
           "input2": [1, 0, 1, 0],
           "output": [1, 1, 1, 0]}

and_dict = {"input1": [1, 1, 0, 0],
            "input2": [1, 0, 1, 0],
            "output": [1, 0, 0, 0]}

xor_dict = {"input1": [1, 1, 0, 0],
            "input2": [1, 0, 1, 0],
            "output": [0, 1, 1, 0]}
```

```python id="58dc2de3-2f78-4e28-aac7-d1431890ffc4"
or_data = pd.DataFrame(data=or_dict)
and_data = pd.DataFrame(data=and_dict)
xor_data = pd.DataFrame(data=xor_dict)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 192} id="d6be8f16-c2ee-4fe3-89e0-fa93b66a8998" outputId="851f19ed-b7f9-4474-a2bc-d3ce3db3967e"
print("OR DataFrame:")
or_data
```

```python colab={"base_uri": "https://localhost:8080/", "height": 192} id="ba3cb10b-8b36-403b-ab2c-00778c221864" outputId="6143e924-bf69-4757-e93e-d3f38a3956a9"
print("AND DataFrame:")
and_data
```

```python colab={"base_uri": "https://localhost:8080/", "height": 192} id="d55b8176-bccc-4047-8af6-86e7e0350410" outputId="ee0fb78a-24c8-434a-f8a2-bfeebb823843"
print("XOR DataFrame:")
xor_data
```

<!-- #region id="3a864cd1-0c17-4c0d-8f44-6e732fdc59ed" -->
## **Plot the DataFrames**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="fe818bbe-60ba-4eca-a121-0ab9ba1b7223" outputId="77706cf4-81cd-4467-e624-bc94fbd4f55e"
fig, axes = plt.subplots(ncols=3, figsize=(14, 4))
axes = axes.flatten()

dataframes = [or_data, and_data, xor_data]
plot_titles = ["OR", "AND", "XOR"]

for i in range(len(dataframes)):
    data = dataframes[i]
    sns.scatterplot(x=data["input1"], y=data["input2"], hue=data["output"], ax=axes[i])
    axes[i].set_title(plot_titles[i])
```

<!-- #region id="f9c6f4a9-f899-4801-9105-f33b49b5b990" -->
## **Train Perceptron Models**
<!-- #endregion -->

```python id="7e557e98-e4bf-4c68-81c0-2e9c80cc9b9a"
# Instantiate 3 different classifiers for 3 datasets
or_clf = Perceptron()
and_clf = Perceptron()
xor_clf = Perceptron()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 74} id="26357aa8-0000-4e60-8d85-c1a87e1e2e72" outputId="650aed00-e73b-4daa-e089-507c81fb4f23"
# Fit the data
or_clf.fit(or_data.iloc[:, :2], or_data.iloc[:, -1])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 74} id="aab5f37d-6a3d-4bd9-8f3b-cc56cab47eda" outputId="3732077f-f67f-48e2-948a-4c385f04c3f6"
and_clf.fit(and_data.iloc[:, :2], and_data.iloc[:, -1])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 74} id="62343b05-9076-483b-93d4-c7bd6712611f" outputId="a670c5c0-32ba-4513-81a0-738184b3c774"
xor_clf.fit(xor_data.iloc[:, :2], xor_data.iloc[:, -1])
```

<!-- #region id="0dd6734f-3634-4b83-8bad-d12c00f1bf1d" -->
## **Check the Coefficients and Intercept**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="44d13821-a1a6-4318-8c1c-47ba4e8a2fbc" outputId="1e14ade2-8054-4941-a3e4-8c95f00445fb"
print("Coeffients (OR Classifier):", or_clf.coef_)
print("Coeffients (AND Classifier):", and_clf.coef_)
print("Coeffients (XOR Classifier):", xor_clf.coef_)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ab33ef7f-09ed-4328-a088-9141defe972a" outputId="216c8c10-3224-4012-a82a-211c8b6ce97c"
print("Intercept (OR Classifier):", or_clf.intercept_)
print("Intercept (AND Classifier):", and_clf.intercept_)
print("Intercept (XOR Classifier):", xor_clf.intercept_)
```

<!-- #region id="fb0df213-9f9f-46e1-b7c5-6c425b7a045e" -->
## **Plot the Decision Boundaries of the Models**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="14833442-82e3-416e-a8e7-cd18a1cf4dd3" outputId="47144ad6-fd70-47aa-bdae-849475e4915e"
classifiers = [or_clf, and_clf, xor_clf]

fig, axes = plt.subplots(ncols=3, figsize=(14, 4))
axes = axes.flatten()

for index, clf in enumerate(classifiers):
    data = dataframes[index]
    slope = clf.coef_.flatten()[0] / clf.coef_.flatten()[1]
    intercept = clf.intercept_[0] / clf.coef_.flatten()[1]

    X = np.linspace(0, 1, 5)
    y = -X * slope - intercept

    # sns.lineplot(x, y, ax=axes[index])
    axes[index].plot(X, y, color="red")
    sns.scatterplot(x=data["input1"], y=data["input2"], hue=data["output"], ax=axes[index])
    axes[index].set_title(f"Decision Boundary of {plot_titles[index]} Classifier")
```

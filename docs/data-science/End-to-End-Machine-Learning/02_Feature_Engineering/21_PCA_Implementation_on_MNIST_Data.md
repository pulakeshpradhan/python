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
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Machine-Learning/blob/main/02_Feature_Engineering/21_PCA_Implementation_on_MNIST_Data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="mzA_8O4xP4GB" -->
# **PCA Implementation on MNIST Data**
<!-- #endregion -->

<!-- #region id="Y7SFTRVILjUO" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="u79cl7QgLyUC" outputId="36518ce7-ff09-44a0-e052-6576b00d4388"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="f0mGFygaL9iL"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="dtcaYZ8QMOqO" -->
## **Read the Data from Kaggle**
<!-- #endregion -->

```python id="vIfnrPBxMR01"
%mkdir ~/.kaggle
%cp kaggle.json ~/.kaggle/
```

```python colab={"base_uri": "https://localhost:8080/"} id="jonjhLq-NWiE" outputId="44913ebd-550a-46c5-cfa0-22eb259cd142"
!kaggle datasets download -d animatronbot/mnist-digit-recognizer
```

```python id="CcGck5o8Nosf"
# Extract the data from Zipfile
import zipfile
zipref = zipfile.ZipFile("/content/mnist-digit-recognizer.zip")
zipref.extractall("/content")
zipref.close()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 273} id="MKrZPx0cUnJ9" outputId="6d30b3a3-b16b-427d-f1c7-35888b24c88c"
# Read the data
df = pd.read_csv("/content/train.csv")
print(df.shape)
df.head()
```

<!-- #region id="cFjxWMTDU9oA" -->
## **Plot the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="j6989QeVWcrR" outputId="f938abf9-321d-46c3-cfaa-09cfefd529b4"
plt.imshow(df.iloc[3][1:].values.reshape(28, 28))
plt.show()
```

<!-- #region id="nDu2dvG9WyU0" -->
## **Train Test Split**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CjD7cQYGW1tl" outputId="82b94fa3-1166-4307-c52b-5cf3bb51ff41"
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:],
                                                    df.iloc[:, 0],
                                                    test_size=0.3,
                                                    random_state=42)
X_train.shape, X_test.shape
```

<!-- #region id="Ojj_CzHAbnC1" -->
## **Apply K-Nearest Neighbour Classifier**
<!-- #endregion -->

```python id="aQ50bUY0sAfl"
# Apply Standardization
scaler = StandardScaler()

# Fit and transform the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 74} id="7nWu7PIkcE4I" outputId="1c17c325-cf2b-46a5-b323-84e8d891cfa8"
# Instantiate a 'KNeighborsClassifier' object
knn = KNeighborsClassifier()

# Fit the training data
knn.fit(X_train_scaled, y_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="yAFn0_KxccCC" outputId="a53eff32-7f51-4ff3-e35a-02669fc35f6c"
# Predict the test data
start = time.time()
y_pred = knn.predict(X_test_scaled)
print("Time taken:", time.time() - start, "Sec")
```

```python colab={"base_uri": "https://localhost:8080/"} id="wPex2_BndFEW" outputId="5edf9756-4d5c-473d-d431-acde22e8aeea"
# Check the accuracy score
accuracy_score(y_test, y_pred).round(2)
```

<!-- #region id="fbmvpDuNdT9m" -->
## **Apply PCA before Classification**
<!-- #endregion -->

```python id="pJbjoSnhdZe-"
# Instantiate a PCA object
pca = PCA(n_components=200)

# Fit and transform the training data
X_train_transformed = pca.fit_transform(X_train_scaled)
X_test_transformed = pca.transform(X_test_scaled)
```

```python colab={"base_uri": "https://localhost:8080/"} id="YYHc0uYCeQpt" outputId="9eb3738e-9cb3-48b5-c91b-b16f1c836af3"
X_train_transformed.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 74} id="n-NUwYQPe7NA" outputId="b3115f2b-62ac-448d-ad0c-0181b0f11b64"
# Instantiate a 'KNeighborsClassifier' object
knn = KNeighborsClassifier()

# Fit the transformed training data
knn.fit(X_train_transformed, y_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="s1o5JOMVfUmY" outputId="dc5c70ed-c975-4eaa-9f21-9bbec4c875fb"
# Predict the transformed test data
start = time.time()
y_pred = knn.predict(X_test_transformed)
print("Time taken:", time.time() - start, "Sec")
```

```python colab={"base_uri": "https://localhost:8080/"} id="ryIPVyaxfcXk" outputId="fb350843-16e6-47df-b153-87ba87b4e1c5"
# Check the accuracy score
accuracy_score(y_test, y_pred).round(2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="L7WlaKLmudj8" outputId="85b6c621-8362-4235-ecc1-55cb3df4d703"
X_train_scaled.shape
```

```python id="Yf16L6gngMrS"
# # Store the accuracy based on number of principal components in a dictionary
# # **This process might take time**
# accuracy_dict = {}

# for i in range(1, 785):
#     print("No. of PC:", i)

#     pca = PCA(n_components=i)

#     X_train_transformed = pca.fit_transform(X_train_scaled)
#     X_test_transformed = pca.transform(X_test_scaled)

#     knn = KNeighborsClassifier()
#     knn.fit(X_train_transformed, y_train)

#     y_pred = knn.predict(X_test_transformed)
#     accuracy = accuracy_score(y_test, y_pred)

#     accuracy_dict[i] = accuracy
```

```python id="BJXXotZ3vUsc"
# sns.lineplot(pd.Series(accuracy_dict), marker="o")
# plt.grid()
# plt.xlabel("No. of Principal Components")
# plt.ylabel("Accuracy")
# plt.show()
```

<!-- #region id="bNQ65Gyqv5c3" -->
## **Data Visualization**
<!-- #endregion -->

<!-- #region id="40thk0pF02RQ" -->
### **2D Visualization**
<!-- #endregion -->

```python id="Xkc4iqSWwGUa"
# Transforming the data into a 2D coordinate system
pca = PCA(n_components=2)

X_train_transformed = pca.fit_transform(X_train_scaled)
X_test_transformed = pca.transform(X_test_scaled)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZD0G93aWwhbb" outputId="0061e8d5-2123-4072-cf20-e3beeb1acc1a"
X_train_transformed
```

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="Ec0HJx89wzP2" outputId="8ad3b194-069e-4d06-b208-24dca3610afd"
fig = px.scatter(x=X_train_transformed[:, 0],
                 y=X_train_transformed[:, 1],
                 color=y_train.astype("str"),
                 color_discrete_sequence=px.colors.qualitative.G10)

fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='black')))

fig.show()
```

<!-- #region id="CsjYIm3X0-VB" -->
### **3D Visualization**
<!-- #endregion -->

```python id="uYhsAJfT1B7B"
# Transforming the data into a 2D coordinate system
pca = PCA(n_components=3)

X_train_transformed = pca.fit_transform(X_train_scaled)
X_test_transformed = pca.transform(X_test_scaled)
```

```python colab={"base_uri": "https://localhost:8080/"} id="LuZ014XX1cgN" outputId="78e3eda6-6276-4b56-fcc3-c759027172d0"
X_train_transformed
```

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="pplih8Tr1m9d" outputId="463d2a21-ac26-4077-b4bd-ff29fb24aecf"
fig = px.scatter_3d(x=X_train_transformed[:, 0],
                    y=X_train_transformed[:, 1],
                    z=X_train_transformed[:, 2],
                    color=y_train.astype("str"),
                    color_discrete_sequence=px.colors.qualitative.G10)

fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color="black")))
fig.update_layout(width=800, height=600)

fig.show()
```

<!-- #region id="bknSVNKzNFoZ" -->
## **Check Eigen Values and Eigen Vectors**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="REhyxBFXMHei" outputId="a4b4de74-a00a-446d-c15f-8cc2e83f0d35"
# Print the Eigen Values
print("Eigen Values:")
pca.explained_variance_
```

```python colab={"base_uri": "https://localhost:8080/"} id="wgpNYO96Mib9" outputId="b532ff57-f4f2-4c26-9d0d-59fc529bdb27"
# Print the Eigen Vectors
print("Eigen Vectors:")
print(pca.components_.shape)
pca.components_
```

<!-- #region id="LNLBlS0QNOnz" -->
## **Find Optimum Number of Principal Components**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pkMbbj_WNT7T" outputId="9ed376c6-5194-4366-b4ca-8b5b5b374ffb"
# Check the percent of variance explained by first 3 principal components
pca.explained_variance_ratio_
```

```python id="E1cnUczDODIa"
# Apply PCA
pca = PCA(n_components=None)

X_train_transformed = pca.fit_transform(X_train_scaled)
X_test_transformed = pca.transform(X_test_scaled)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="ewPUzJonOOIb" outputId="44d7fa9e-a0c3-4647-d2b8-4145a9857047"
sns.lineplot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("No. of Principal Components")
plt.ylabel("Cumulative Variance")
plt.grid()
plt.show()
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/20_PCA_Step_by_Step.ipynb)

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

<!-- #region id="rvQCSxuiyMNq" -->
# **Principal Component Analysis (PCA): Step by Step**

PCA, or Principal Component Analysis, is a dimensionality reduction technique commonly used in machine learning and statistics. Its primary purpose is to transform high-dimensional data into a lower-dimensional form while retaining as much of the original data's variance as possible. This is achieved by identifying the principal components, which are linear combinations of the original features that capture the most significant sources of variation in the data.

Here's a brief overview of the PCA process:

1. **Mean Centering:** The mean of each feature is subtracted from the data to center it around the origin.

2. **Covariance Matrix Calculation:** The covariance matrix of the mean-centered data is computed. This matrix describes the relationships between different features and their variances.

3. **Eigendecomposition:** The next step involves finding the eigenvectors and eigenvalues of the covariance matrix. Eigenvectors represent the directions of maximum variance, and eigenvalues indicate the magnitude of variance in those directions.

4. **Selection of Principal Components:** The eigenvectors are ranked by their corresponding eigenvalues, and the top k eigenvectors (where k is the desired lower dimensionality) are selected to form a new matrix, known as the transformation matrix.

5. **Projection:** The original data is then projected onto this lower-dimensional subspace using the transformation matrix, resulting in a new set of features called principal components.

PCA is widely used for various purposes, including data compression, noise reduction, visualization, and as a preprocessing step in machine learning tasks to enhance model performance by reducing the dimensionality of the input data.
<!-- #endregion -->

<!-- #region id="r6LQzG2Vy-Lm" -->
## **Import Required Libraries**
<!-- #endregion -->

```python id="AZ48CytAykB6"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="ix_eNw7Vy9tj" -->
## **Make the Data for Classification**
<!-- #endregion -->

```python id="_jzvf4DYzISk"
# Create a data with 3 features
X, y = make_classification(
    n_samples=200,  # Total number of samples
    n_features=3,   # Number of features
    n_informative=2,  # Number of informative features
    n_redundant=0,    # Number of redundant features
    n_classes=2,      # Number of classes (binary classification)
    random_state=42   # Random seed for reproducibility
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 223} id="vjWyOItHz786" outputId="43899479-edc0-4c13-fd81-38a860b01ed7"
# Convert the data array into pandas dataframe
df = pd.DataFrame({"feature1": X[:, 0], "feature2": X[:, 1], "feature3": X[:, 2], "target": y})
print(df.shape)
df.head()
```

<!-- #region id="zm3nzIja0eoe" -->
## **Plot the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="ZfXURmB40nBu" outputId="c0a92bac-515f-4d95-8892-a23b1b249fdc"
# Plot a 3D scatter plot
fig = px.scatter_3d(df, x=df["feature1"], y=df["feature2"], z=df["feature3"],
                    color=df["target"].astype("str"), width=600, height=600)
fig.update_traces(marker=dict(size=4, line=dict(width=2, color="DarkSlateGrey")),
                  selector=dict(mode="markers"))
fig.show()
```

<!-- #region id="oqbXdfLG2Mb0" -->
## **Step-1: Mean Centering**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="Aq-R4ul32TCh" outputId="e087f49f-4193-4415-98a4-f4c340e2ce8e"
# Instantiate a StandardScaler object
scaler = StandardScaler()

# Fit and transform input features
X_scaled = scaler.fit_transform(df.iloc[:, :-1])

# Convert the scaled array into pandas dataframe
X_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])
X_scaled.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="K8nnzj7421K9" outputId="1b8ac7b8-dd27-4763-e448-a29381d69529"
X_scaled.describe().round(2)
```

<!-- #region id="wzJ3bmkM3S54" -->
## **Step-2: Covariance Matrix Calculation**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 161} id="lymj5-u93aGi" outputId="fba1d81a-d637-42d8-eb33-652e717bf6d2"
covariance_matrix = X_scaled.cov()
print("Covariance Matrix:")
covariance_matrix
```

<!-- #region id="UgD6ffHM347m" -->
## **Step-3: Eigendecomposition**
<!-- #endregion -->

```python id="lgJhXRSH4A3e"
# Calculate the Eigen Vectors and Eigen Values of the Variance-Covariance matrix
eigen_values, eigen_vectors = np.linalg.eig(np.array(covariance_matrix))
```

```python colab={"base_uri": "https://localhost:8080/"} id="fqPANKey4q8j" outputId="65fbd9f6-215d-4d4c-eafe-a852a561ec4c"
eigen_values
```

```python colab={"base_uri": "https://localhost:8080/"} id="WyuQDa354sIR" outputId="93ac37b9-a579-4c88-8ac4-2529670b33ac"
eigen_vectors
```

<!-- #region id="vMlVcA00KZ4P" -->
## **Step-4: Selection of Principal Components**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="2JffaK1TK0_F" outputId="598728d9-b687-4c29-eaad-4f1675255e14"
# Plot the eigen vectors in 3D space
# Create a 3D scatter plot
i = 1

fig = go.Figure()

for eig_vector in eigen_vectors:
    eig_vector = np.hstack((np.array([0, 0, 0]), eig_vector))
    fig.add_trace(go.Scatter3d(
        x=[eig_vector[0], eig_vector[3]],
        y=[eig_vector[1], eig_vector[4]],
        z=[eig_vector[2], eig_vector[5]],
        mode='lines+markers',
        marker=dict(size=6),
        line=dict(width=4),
        name='Eigen Vector ' + str(i)
    ))

    i+=1

fig.add_trace(go.Scatter3d(
    x=X_scaled.iloc[:, 0],
    y=X_scaled.iloc[:, 1],
    z=X_scaled.iloc[:, 2],
    mode='markers',
    marker=dict(size=3, color=df["target"], colorscale='RdYlGn',
                opacity=0.4, line=dict(width=1, color="black")),
    name='Scatter Points'
))

# Set axis labels
fig.update_layout(scene=dict(xaxis_title='feature1', yaxis_title='feature2', zaxis_title='feature3'), width=800, height=600)

# Show the interactive plot
fig.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="5Jv2sj5QXKCl" outputId="037c1a6a-2e9e-4bbd-841c-d9df9cfa7689"
# Get the sorted indices that would sort eigen values in descending order
sorted_indices = np.argsort(eigen_values)[::-1]
sorted_indices
```

```python colab={"base_uri": "https://localhost:8080/"} id="9QXBZ4l_ZXLU" outputId="02f717a3-7d69-4e57-9fb8-db5ba1b21ece"
# Select top two principal components
pc = eigen_vectors[sorted_indices[:2]]
pc
```

<!-- #region id="P33puAJ_Zsxu" -->
## **Step-5: Projection**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="ldYrzZKvZ6ZZ" outputId="60f26837-d996-4a5a-fcf7-34c96add27f4"
# Project the scaled data onto principal components
transformed_df = np.dot(X_scaled, pc.T)
new_df = pd.DataFrame(transformed_df, columns=["PC1", "PC2"])

# Add the 'target' column
new_df["target"] = df["target"].values
new_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="e97Bbzu3a4Hf" outputId="b68421c4-9a6e-49ca-d76e-c2cd695eaf24"
sns.scatterplot(x=new_df["PC1"], y=new_df["PC2"], c=new_df["target"], cmap="RdYlGn")
plt.title("Scatterplot between PC1 and PC2")
plt.show()
```

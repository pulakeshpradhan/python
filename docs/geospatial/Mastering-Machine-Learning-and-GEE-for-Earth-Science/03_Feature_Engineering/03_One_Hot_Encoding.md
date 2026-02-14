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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/03_Feature_Engineering/03_One_Hot_Encoding.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="cea85c4b-1cba-48b2-a46c-7f6078ff6b16" -->
# **One Hot Encoding**
One-Hot Encoding is a popular technique used in machine learning and data preprocessing, especially when dealing with categorical data. It is used to represent categorical variables as binary vectors or matrices, where each category is mapped to a unique binary value.

This transformation is necessary because many machine learning algorithms and models require numerical input, and categorical data in its raw form cannot be directly used in these algorithms.
<!-- #endregion -->

<!-- #region id="3ad91c14-7815-4553-a1ea-2eba45fcc5bb" -->
<img src="https://miro.medium.com/v2/resize:fit:1358/1*ggtP4a5YaRx6l09KQaYOnw.png" width="90%">
<!-- #endregion -->

<!-- #region id="a592ac47-26e8-4f2f-b8ce-e2aada1a8d5d" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vywY7cnJfjE2" outputId="904d28da-cf60-48fd-a550-9b53317b744d"
from google.colab import drive
drive.mount("/content/drive/")
```

```python id="8df890ce-c77a-4a58-aa7f-cbe26712a7e6"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
```

<!-- #region id="bc42bfd3-0e23-40e9-b5a4-9c7f3ab27ac9" -->
## **Read the Data**
<!-- #endregion -->

```python id="5ecc074a-8716-404b-93a0-0e56f39cd863" outputId="3ff06609-9786-43b3-d0d8-b5914e290e18" colab={"base_uri": "https://localhost:8080/", "height": 206}
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/cars.csv")
df.head()
```

```python id="3d98f302-de65-467a-ac5a-28235fd04d5c" outputId="2a5e53a0-1868-49e3-ea5e-dbcc35d03c7a" colab={"base_uri": "https://localhost:8080/"}
df.shape
```

```python id="8d7fc924-1386-481f-88a4-12f742a88cb4" outputId="f4229462-69bc-4c66-ed7f-e2f17a40c019" colab={"base_uri": "https://localhost:8080/"}
# Check the number of unique brand names
df["brand"].nunique()
```

```python id="4f084779-4a43-4e4b-ae62-73a5a2aa8d46" outputId="b1fc186f-a787-4e26-a112-33758dae8ce0" colab={"base_uri": "https://localhost:8080/"}
# Count the values for each brand in 'brand' column
df["brand"].value_counts()
```

```python id="1e2ea2a1-a7da-440e-aaf9-ea9f114b64b7" outputId="1b8b131e-a6f1-433e-824c-bee284a41caa" colab={"base_uri": "https://localhost:8080/"}
# Count the values for each unique name in 'fuel' column
df["fuel"].value_counts()
```

```python id="0fa49577-16e8-4650-906b-8d93a9ddbcba" outputId="3f828a5d-856b-4858-e3f6-05fa6aafd018" colab={"base_uri": "https://localhost:8080/"}
# Count the values for each unique name in 'owner' column
df["owner"].value_counts()
```

<!-- #region id="b3b8edde-d01a-4ad0-8d3e-86c924e70699" -->
## **One Hot Encoding with Pandas**
<!-- #endregion -->

```python id="6294b599-a14c-46d1-9669-0e975421e300" outputId="aee4fc44-4896-4c95-844d-18635d8a38ff" colab={"base_uri": "https://localhost:8080/", "height": 478}
# Applying One Hot Encoding on 'fuel' and 'owner' columns
pd.get_dummies(data=df, columns=["fuel", "owner"])
```

<!-- #region id="68f66cd2-69c2-4284-bafc-e1993c5d0fee" -->
## **K-1 One Hot Encoding with Pandas**
<!-- #endregion -->

<!-- #region id="8106db49-aec3-4092-b4fa-617611c66faa" -->
When using the `pd.get_dummies()` function in Pandas, you can drop the first category (column) of each categorical variable to avoid multicollinearity, which can be useful in certain situations. This is done using the `drop_first` parameter. Setting `drop_first=True` will drop the first category from each categorical variable after one-hot encoding.
<!-- #endregion -->

```python id="54d231b2-7b77-4c65-81f5-283f6d828955" outputId="8baf4391-e7d2-467d-ba30-eadeaacc2ba5" colab={"base_uri": "https://localhost:8080/", "height": 441}
# Applying One Hot Encoding on 'fuel' and 'owner' columns
# Removing the first categorical variable to avoid multicolinearity
pd.get_dummies(data=df, columns=["fuel", "owner"], drop_first=True)
```

<!-- #region id="f3acc4f8-dfed-422a-a3c3-ab25c2b7b363" -->
## **One Hot Encoding using Sklearn**
<!-- #endregion -->

<!-- #region id="85e1bb8f-f658-4014-84f1-e1995b4cc55d" -->
### **Train Test Split**
<!-- #endregion -->

```python id="b34cc2a2-7692-4c3c-8420-0a2d9f69181a" outputId="11c067ad-8f9b-4466-c12a-0a20bf2071b9" colab={"base_uri": "https://localhost:8080/", "height": 206}
# Print the dataframe
df.head()
```

```python id="d5d1d1f0-21f4-49a0-a2bd-ffc47e1e9e8b" outputId="2da27c1a-6ec4-48bc-ac3c-4e17f86c450e" colab={"base_uri": "https://localhost:8080/"}
X_train, X_test, y_train, y_test = train_test_split(df.drop("selling_price", axis=1),
                                                    df["selling_price"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, y_train.shape
```

```python id="16f19237-de20-4b22-840e-c26e6f1d1d84" outputId="47dfdab0-40e2-4802-bb25-ffda83530463" colab={"base_uri": "https://localhost:8080/", "height": 206}
X_train.head()
```

```python id="98044aa4-1fbb-4bd5-930d-3229f20f2423" outputId="bb8d51a4-69be-492e-98ad-41b0f4bf58f3" colab={"base_uri": "https://localhost:8080/"}
X_train.shape
```

<!-- #region id="1de9a254-70a8-4b1a-b71d-4d980643559f" -->
### **Apply OHE on 'fuel' and 'owner' Columns**
<!-- #endregion -->

<!-- #region id="Ccyow8j-t0Nz" -->
🤔 **Note:** The `sparse_output` parameter in the `OneHotEncoder` controls the format of the output matrix. When `sparse_output` is set to `True`, the output matrix will be represented as a sparse matrix (SciPy sparse matrix format), which is a memory-efficient way to store matrices with a large number of zero values. This can be beneficial when dealing with high-dimensional one-hot encoded matrices, as it saves memory compared to using a dense matrix representation. On the other hand, when `sparse_output` is set to `False` (the default), the output matrix will be a dense NumPy array.
<!-- #endregion -->

```python id="7af538f2-16c0-4cd9-92d1-fc75e7d92c30" outputId="998893dd-0bde-45b2-f239-11e93c33a311" colab={"base_uri": "https://localhost:8080/"}
# Creating an object of the One Hot Encode class
one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False, dtype=np.int8)

# Separating the 'fuel' and 'owner' columns from the X_train dataframe
# Fit the separated training data
one_hot_encoder.fit(X_train[["fuel", "owner"]])

# Transform the separated training data
X_train_encoded = one_hot_encoder.transform(X_train[["fuel", "owner"]])
X_train_encoded
```

```python id="c9c9a5e4-3584-4786-8450-f0e26964cbb8" outputId="15edf17e-dd7c-4aa8-ea8b-fad7d8fdaf2d" colab={"base_uri": "https://localhost:8080/"}
X_train_encoded.shape
```

```python id="56c09bbd-14fd-420f-9ecb-07656f0697b0" outputId="d228d914-5afa-493b-c639-4a0d72eee8a5" colab={"base_uri": "https://localhost:8080/"}
# Merge the X_train_encoded columns with the 'brand' and 'km_driven' columns
X_train_merged = np.hstack((X_train[["brand", "km_driven"]], X_train_encoded))
X_train_merged
```

```python id="bfd54d97-10f0-4c3a-99ec-80c13a79cdb7" outputId="f22d45f0-ffd0-462a-ba51-ed8d44d71eb7" colab={"base_uri": "https://localhost:8080/"}
X_train_merged.shape
```

```python id="a1e833ce-9ac8-4b8e-abbe-af968a4c248b" outputId="98260aa8-185c-45a9-b95d-3d2562d20086" colab={"base_uri": "https://localhost:8080/"}
# Print the column names of the encoded x_train data
one_hot_encoder.get_feature_names_out()
```

```python id="3dfcee27-04be-47ec-8167-e9fcfe8d1611" outputId="ef06429f-b9a2-4bc2-f93c-3c94ba0c27f5" colab={"base_uri": "https://localhost:8080/"}
# Define the column names in an array
column_names = np.concatenate((X_train.columns[0:2], one_hot_encoder.get_feature_names_out()), axis=0)
print(len(column_names))
column_names
```

```python id="9c91d3b0-07e1-4897-9e39-a0ba836780cb" outputId="4aedf6da-7f34-482e-8510-504bee5848cc" colab={"base_uri": "https://localhost:8080/", "height": 441}
# Convert the x_train_merged array into pandas dataframe
X_train_encoded = pd.DataFrame(data=X_train_merged, columns=column_names)
X_train_encoded
```

```python id="de07eec4-e6ec-4b03-9fdb-2e60d34204c1" outputId="b9f07f80-47bd-4aaf-ad40-d4b732db111e" colab={"base_uri": "https://localhost:8080/"}
X_train_encoded.shape
```

```python id="7c94761a-6567-4350-a2e4-5286efef176a" outputId="2f516e6e-3cbe-40a2-9a72-11bde6342c4f" colab={"base_uri": "https://localhost:8080/", "height": 206}
# Print the x_test data
X_test.head()
```

```python id="c5639c46-4bc4-43cf-ab73-4677391bc99e" outputId="7251cb8a-03f5-4485-9b11-d1e04f665ec6" colab={"base_uri": "https://localhost:8080/"}
# Encode x_test data
X_test_encoded = one_hot_encoder.transform(X_test[["fuel", "owner"]])
X_test_encoded
```

```python id="d7c02894-d6c8-4f86-aa7b-6e5ab4a33c2d" outputId="2dcb6c68-9aac-4721-fe32-52019580e9e7" colab={"base_uri": "https://localhost:8080/"}
# Merge the x_test_encoded columns with the 'brand' and 'km_driven' columns
X_test_merged = np.hstack((X_test.iloc[:, 0:2], X_test_encoded))
X_test_merged
```

```python id="6f6d04cc-35f1-4150-8f7d-42e600151da6" outputId="986b8b4e-5385-42ea-988c-2cb47caef826" colab={"base_uri": "https://localhost:8080/", "height": 441}
# Convert the x_test_merged array into pandas dataframe
X_test_encoded = pd.DataFrame(data=X_test_merged, columns=column_names)
X_test_encoded
```

<!-- #region id="469184eb-8967-42b9-bbdd-b63fc1829971" -->
## **Apply OHE on 'brand' Column using Pandas**
<!-- #endregion -->

```python id="0a01b063-da22-4738-acf1-60a05856c66d" outputId="c73fd51d-db7f-4dec-9e9a-4993de415991" colab={"base_uri": "https://localhost:8080/"}
# Count the values for each brand in 'brand' column
counts = df["brand"].value_counts()
counts
```

```python id="be7a1463-81ff-4d9e-af2e-ccf554bdb20e" outputId="5541518a-b922-4ec8-c265-5a5e5e982a1c" colab={"base_uri": "https://localhost:8080/"}
# Check the total number of unique brands
df["brand"].nunique()
```

```python id="8ba47626-20ed-433a-9f47-ca4e0596ba14"
# Define a threshold
threshold = 100
```

```python id="6a186820-a8f7-4229-aed9-c11c43fb902a" outputId="a7c61f7e-78e4-47ab-cfe2-2a469f10c529" colab={"base_uri": "https://localhost:8080/"}
# Store the name of brands in a list where the value count is less than 100
repl = counts[counts <= threshold].index
repl
```

```python id="f4402431-aa57-458f-93fe-bc6505ed6dba" outputId="c61ec684-f221-44ec-8932-8f5ea22de056" colab={"base_uri": "https://localhost:8080/", "height": 423}
# Replace the name of the brand with 'others'
new_df = df.replace(to_replace=repl, value="Others")
new_df
```

```python id="76b41c92-08a4-4d7c-869e-6a4d1fe4ab2a" outputId="e3c2da61-e496-460e-f4af-078bfe04244a" colab={"base_uri": "https://localhost:8080/"}
new_df["brand"].value_counts()
```

```python id="896eb779-396f-4d94-8642-4052dd760309" outputId="6fef356f-8d21-4598-f9b5-389ba4931259" colab={"base_uri": "https://localhost:8080/", "height": 676}
# Apply OHE on 'brand' column of the new dataframe
pd.get_dummies(data=new_df["brand"]).sample(20)
```

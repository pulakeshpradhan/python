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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/03_Feature_Engineering/01_Normalization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="c55279d3-3dd8-4db7-999d-2ab610301732" -->
# **Normalization - Feature Scaling**
Normalization, also known as Min-Max scaling, is a feature scaling technique used in data preprocessing to rescale numerical features to a specific range, typically between 0 and 1. The goal of normalization is to ensure that all feature values have similar scales, making them more suitable for machine learning algorithms that are sensitive to the magnitude of input features.
<!-- #endregion -->

<!-- #region id="87b13764-ce4e-414d-be50-ba7604422c6b" -->
<center><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTBOZqDiNcILVqNqUPmDj8r3I7f_GYD9Op6qrjK1BB8r5iKASSzZoyWOTv4HE4V2JVYGx0&usqp=CAU"></ceneter>
<!-- #endregion -->

<!-- #region id="7226ee7c-3b92-407f-aaea-f433ae579acf" -->
## **When to use Normalization?**
<!-- #endregion -->

<!-- #region id="c43ef524-6f5b-419e-839d-9c04df1fd300" -->
<center><img src="https://miro.medium.com/max/1400/1*qRmiffZgkNaXnTBZwDafCA.png" width="80%"></center>
<!-- #endregion -->

<!-- #region id="a434553c-7be7-4d9e-aabd-457b646966ea" -->
# **Example**
<!-- #endregion -->

<!-- #region id="25b31dea-ee7d-4365-a8f9-0563276a8e23" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="27KLSOV_g-Q4" outputId="6077e63d-ac57-4d59-a86b-fc98e045a465"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="d2ae2692-7dfe-4067-9a4f-27ce661465dd"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```

<!-- #region id="4c24c6ea-4ba5-41ea-951d-9ab9b41ac774" -->
## **Read the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="c52ff947-954e-436f-8b13-17bc76091f95" outputId="525c79a6-0902-4842-bd25-3518b5818d0c"
# Read some specific column from the data
csv_path = r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/wine_data.csv"
df = pd.read_csv(csv_path, usecols=["Class", "Alcohol", "Malic acid"])
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="2a74fec9-2403-429a-9151-36f2ce595fca" outputId="db605697-b94d-4806-e354-b916abb4bf69"
df.shape
```

<!-- #region id="ec0a3a59-3640-4124-a755-b4e33a9e0088" -->
## **Data Visualization**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 487} id="75695559-c12b-43c4-8192-3648a7a18b5b" outputId="a19c21d5-46bc-4382-cc99-3befd9fde03a"
# Creating probabilty density plots of the data
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

sns.kdeplot(data=df["Alcohol"], ax=ax1, c="red", label="Alcohol")
ax1.set_title("Probability Distribution of Alcohol")
ax1.legend()

sns.kdeplot(data=df["Malic acid"], ax=ax2, c="blue", label="Malic acid")
ax2.set_title("Probability Distribution of Malic acid")
ax2.legend()

plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 468} id="0f2a3aa2-b018-46fd-b450-a3cb4418bb7b" outputId="484f74ad-13f8-46e1-cb03-7cd1df62b597"
# Creating a scatter plot of the data
sns.scatterplot(x=df["Alcohol"], y=df["Malic acid"], hue=df["Class"], palette=["red", "blue", "green"])
```

<!-- #region id="45854672-298c-437b-a9c4-aacff619310c" -->
## **Train Test Split**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="d7bbdbed-83ed-40ee-99a1-67763bc3c900" outputId="65dc23ec-b032-40f7-9d9e-0c9a923207b8"
X_train, X_test, y_train, y_test = train_test_split(df.drop("Class", axis=1),
                                                    df["Class"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

<!-- #region id="0cda6d24-1504-4538-9d96-909f838dacf2" -->
## **MinMax Scaler**
The Min-Max Scaler is a feature scaling technique used in machine learning to transform and normalize the values of features within a specific range. It is particularly useful when dealing with features that have varying scales, as it helps ensure that all features contribute equally to the learning process. Min-Max scaling transforms the original values of features to a specified range, typically between 0 and 1.
<!-- #endregion -->

```python id="365d98cf-a27a-4bd8-b1e1-ae22da20479b"
# Creating a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# Transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python id="6ee2db16-6b58-42e5-bff4-806cb84424d8"
# transform always returns a numpy array
# Converting the scaled numpy arrays into pandas dataframes
X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=["Alcohol", "Malic acid"])
X_test_scaled = pd.DataFrame(data=X_test_scaled, columns=["Alcohol", "Malic acid"])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="cKl8RjDWijtV" outputId="329e14d2-3648-4225-c6b5-f848a741b821"
X_train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="b895afde-0b26-4073-b83e-871af4ecbc89" outputId="bd7306e0-0e76-405e-f516-38d4976f741b"
X_train_scaled.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="32e2c777-3ba7-4a3b-9eaa-434070cf0dca" outputId="4510c793-6220-44ac-916a-58516ad20a92"
# Describe the training data
np.round(X_train.describe(), 2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="93541294-b3f9-49b0-86c8-b63d51a74077" outputId="d1324f70-86d4-431c-8799-13f7bdc383f3"
# Describe the scaled training data
np.round(X_train_scaled.describe(), 2)
```

<!-- #region id="b22a40e5-e280-4658-a955-fa6da7f31925" -->
## **Effect of Scaling**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 487} id="fa5097dd-b418-424d-a8a8-cde1931345c1" outputId="d09c531a-d58c-4d7f-b5d4-da07ced2149b"
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a scatter plot of the training data
sns.scatterplot(data=X_train, x="Alcohol", y="Malic acid", ax=ax1, c=y_train)
ax1.set_title("Before Scaling")

# Creating a scatter plot of the scaled training data
sns.scatterplot(data=X_train_scaled, x="Alcohol", y="Malic acid", ax=ax2, c=y_train)
ax2.set_title("After Scaling")

plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 468} id="b333b729-bb11-4537-8d1d-66fa49d2c757" outputId="ade9f8f7-397a-4d89-c247-7edc05358a8f"
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a probability density plot of the training data
sns.kdeplot(data=X_train["Alcohol"], ax=ax1, label="Alcohol")
sns.kdeplot(data=X_train["Malic acid"], ax=ax1, label="Malic acid")
ax1.legend()
ax1.set_xlabel("")
ax1.set_title("Before Scaling")

# Creating a probability density plot of the scaled training data
sns.kdeplot(data=X_train_scaled["Alcohol"], ax=ax2, label="Alcohol")
sns.kdeplot(data=X_train_scaled["Malic acid"], ax=ax2, label="Malic acid")
ax2.legend()
ax2.set_xlabel("")
ax2.set_title("After Scaling")

plt.show()
```

<!-- #region id="f558e0ca-45b9-4307-872f-a94775e1a148" -->
## **Comparison of Distribution**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 487} id="8513abf7-2cd2-461c-b088-65610d0fc78b" outputId="e0f0273e-b09c-4da8-bebe-db0100c07b6a"
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a probabilty density plot of the 'Alcohol' column from the training data
sns.kdeplot(X_train["Alcohol"], ax=ax1)
ax1.set_title("Alcohol Distribution before Scaling")

# Creating a probabilty density plot of the 'Alcohol' column from the scaled training data
sns.kdeplot(X_train_scaled['Alcohol'], ax=ax2)
ax2.set_title("Alcohol Distribution after Scaling")

plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 487} id="738bf89c-356b-4bf5-b8f6-3dbae8032b50" outputId="a0b517f6-9609-49fc-bf37-f6283255419a"
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a probabilty density plot of the 'Malic acid' column from the training data
sns.kdeplot(X_train["Malic acid"], ax=ax1)
ax1.set_title("Malic acid Distribution before Scaling")

# Creating a probabilty density plot of the 'Malic acid' column from the scaled training data
sns.kdeplot(X_train_scaled['Malic acid'], ax=ax2)
ax2.set_title("Malic acid Distribution after Scaling")

plt.show()
```

<!-- #region id="fc249902-a671-4712-af54-d8fa00a2aacf" -->
🤔 **Note:**<br>
**Difference between Standardization and Normalization**<br>
Standardization and normalization are two common techniques for feature scaling in data preprocessing, and they have different approaches and effects on the data:

**Goal:**<br>
* **Standardization:** The goal of standardization is to transform the data in such a way that it has a mean of 0 and a standard deviation of 1. It centers the data around zero and scales it by the standard deviation.

* **Normalization:** The goal of normalization is to rescale the data to a specific range, typically between 0 and 1. It preserves the relative relationships between data points but scales them to fit within the chosen range.

<br>

<center><img src="https://miro.medium.com/v2/resize:fit:744/1*HW7-kYjj6RKwrO-5WTLkDA.png" width="70%"></center>
<!-- #endregion -->

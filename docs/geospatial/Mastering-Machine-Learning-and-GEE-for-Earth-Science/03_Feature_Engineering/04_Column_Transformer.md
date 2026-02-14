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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/03_Feature_Engineering/04_Column_Transformer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="64442240-cb93-463c-85e5-18d2858551e2" -->
# **Column Transformer**
The ColumnTransformer is a feature in scikit-learn, a popular Python machine learning library, that allows you to apply different preprocessing steps to different subsets of the columns (features) in your dataset. It is particularly useful when you have a dataset with a mix of numerical and categorical features, and you want to apply different transformations to these feature types.

Here's an overview of how the ColumnTransformer works:

1. **Specify Transformers:**<br> First, you define a list of transformers, where each transformer specifies a particular preprocessing step to be applied to a subset of the columns. For example, you might have one transformer for numerical columns (e.g., scaling), another for categorical columns (e.g., one-hot encoding), and maybe even other transformers for specific subsets of columns.

2. **Specify Columns:**<br> For each transformer, you also specify which columns it should be applied to. This is done using the columns parameter, where you can specify either column indices or column names.

3. **Combine Transformers:**<br> You create a ColumnTransformer object and pass in the list of transformers. You can also specify what to do with the remaining columns that are not specified in any of the transformers, using the remainder parameter. Options include dropping them or passing them through without any transformation.

4. **Fit and Transform:**<br> You can then fit the ColumnTransformer on your dataset using the fit method, and subsequently transform your dataset using the transform method. The ColumnTransformer applies the specified transformations to the designated columns and returns a transformed dataset.
<!-- #endregion -->

<!-- #region id="0db175b1-f2ff-45c9-b5de-2713416f0bce" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cef0fd0d-7536-4eb0-978b-7323a5417ee9" outputId="56adefd9-b6a3-44c7-dd64-cd6b3993b426"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="e6d0edce-9ca7-47a1-b4ac-efe9ae7c1352"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
```

<!-- #region id="ec9a7778-44b4-4efa-a919-07abf5e984da" -->
## **Read the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="a208816c-5f6e-4b1c-98f2-2188ffb7bcd2" outputId="43109656-d869-4796-f8e9-95b6cb211e49"
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/covid_toy.csv")
df
```

```python colab={"base_uri": "https://localhost:8080/"} id="453d72e1-778d-471c-b001-dab4a238cafd" outputId="2fb127d0-a594-487b-8b23-18ea6d6758fa"
# Check the information of the columns
df.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="c37008a3-8bea-4c9b-a763-b7a82dfda3ed" outputId="cf0da5d6-0704-426e-bedc-ee158fc70c34"
# Check the number of null values in each column
df.isnull().sum()
```

```python colab={"base_uri": "https://localhost:8080/"} id="4d5b0d2b-c706-456b-85ab-5a1992708e81" outputId="2efdf8f1-8b25-4a7a-dda1-6948fe7328a2"
# Check all the unique values of the categorical columns
for column in df.select_dtypes(include="object").columns:
    unique_values = df[column].unique()
    print(f"{column}: {unique_values}")
```

<!-- #region id="d781b9b7-5b5b-4b7f-ab4d-dc0cab0b1e6d" -->
## **Preprocessing without `ColumnTransformer`**
<!-- #endregion -->

<!-- #region id="a960625a-b844-422e-bf5d-06d3034dc370" -->
### **Train Test Split**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4ed2d3e7-0f3c-43cb-8479-eacf4a2c688b" outputId="1777e94a-5c44-4abe-ec80-9568d7915b88"
X_train, X_test, y_train, y_test = train_test_split(df.drop("has_covid", axis=1),
                                                    df["has_covid"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 363} id="e9e080a0-7f73-42f2-af69-0443b6a60931" outputId="959167dd-8ba1-42f0-b506-d9256c2cc416"
X_train.head(10)
```

<!-- #region id="fc40ae23-ae5e-4e0b-9595-0d2430a834d5" -->
### **Fill the Null Values of `fever` Column using `SimpleImputer`**

- **SimpleImputer:**<br>
 It is an univariate imputer for filling missing values with simple strategies. It replaces missing values using a descriptive statistic (e.g. mean, median, or most frequent) along each column, or using a constant value.
<!-- #endregion -->

```python id="f13c1455-6d3e-42e7-9ed9-c3404ae9c0ba"
# Create a SimpleImputer object
simple_imputer = SimpleImputer(strategy='mean')

# Fit the 'fever' column of the training data
simple_imputer.fit(X_train[["fever"]])

# Transform the 'fever' column of the training and testing data
X_train_fever = simple_imputer.transform(X_train[["fever"]])
X_test_fever = simple_imputer.transform(X_test[["fever"]])
```

```python colab={"base_uri": "https://localhost:8080/"} id="655d6e9c-374c-4605-b825-95dafb9f3a80" outputId="ef0116d9-b4bf-490f-97ec-245182ef2a99"
# Print the first ten values of the x_train_fever
X_train_fever[:10]
```

<!-- #region id="c5a1e36f-e94f-4c49-aefe-059f4160d773" -->
### **Apply `OrdinalEncdoer` to `cough` Column**


<!-- #endregion -->

```python id="7bdca428-4b99-4b01-957d-a2a4c1263846"
# Create an object of the OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[["Mild", "Strong"]], dtype=int)

# Fit the 'cough' column of the training data
ordinal_encoder.fit(X_train[["cough"]])

# Transform the 'cough' column of the training and testing data
X_train_cough = ordinal_encoder.transform(X_train[["cough"]])
X_test_cough = ordinal_encoder.transform(X_test[["cough"]])
```

```python colab={"base_uri": "https://localhost:8080/"} id="412fb5d2-8c09-4160-a684-1250d01d5fa1" outputId="fd3c79bb-9c1f-429d-cc5b-c9b55b2fff28"
# Print the first ten values of the x_train_cough
X_train_cough[:10]
```

<!-- #region id="ccc958a8-7f80-4e0c-8596-dad4dae1b63e" -->
### **Apply `OneHotEncdoer` to `gender` and `city` Columns**
<!-- #endregion -->

```python id="04163d54-e874-4a67-bece-fe6af0a45c97"
# Create an object of the OneHotencoder
one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False, dtype=int)

# Fit the 'genedr' and 'city' columns of the training data
one_hot_encoder.fit(X_train[["gender", "city"]])

# Transform the 'genedr' and 'city' columns of the training and testing data
X_train_gender_city = one_hot_encoder.transform(X_train[["gender", "city"]])
X_test_gender_city = one_hot_encoder.transform(X_test[["gender", "city"]])
```

```python colab={"base_uri": "https://localhost:8080/"} id="2a57d9c3-e81a-4848-9a3f-c661f5110c69" outputId="38d0a748-838d-4a8c-e3aa-090b300b1055"
# Check the new column names after applying One Hot Encoding
one_hot_encoder.get_feature_names_out()
```

```python colab={"base_uri": "https://localhost:8080/"} id="1592c5ac-1b95-46df-899a-04fd7c68c211" outputId="c540245b-06eb-490d-89f9-a38cc1bd2aaa"
# Print the first ten values of the x_train_gender_city
X_train_gender_city[:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="6ed46656-b7af-4e36-9764-c7338758d148" outputId="1a62d012-0053-4f18-e5f8-eaf2b4cbd59b"
X_train_cough.shape
```

```python id="82b8cbd3-ff65-47e8-9a91-1ab8e6857925"
# Convert the 'age' column into numpy array
X_train_age = np.array(X_train["age"]).reshape((70, 1))
X_test_age = np.array(X_test["age"]).reshape((30, 1))
```

```python colab={"base_uri": "https://localhost:8080/"} id="7f2962fd-4c3c-4b4c-a174-85c92aafae24" outputId="ab6a4655-183b-4e99-ca71-823d12aad734"
# Print the first ten values of the x_train_age
X_train_age[:10]
```

<!-- #region id="a02d06ac-0104-41c2-8dc5-a5d5cf022543" -->
### **Concatenating all the Arrays for the Training and Testing Data**
<!-- #endregion -->

```python id="6bbc7b84-3409-426a-992a-bfe249780de7"
# Concatenating all the columns of the training data
X_train_transformed = np.concatenate((X_train_age, X_train_fever, X_train_cough, X_train_gender_city), axis=1)

# Concatenating all the columns of the training data
X_test_transformed = np.concatenate((X_test_age, X_test_fever, X_test_cough, X_test_gender_city), axis=1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="30eb309f-e6fc-4a75-b484-451dcb931574" outputId="d3444b87-1b17-4498-d4cf-32bb8a1a79e1"
# Defining the column names of the transformed dataframe
column_names = np.concatenate((np.array(["age", "fever", "cough"]), one_hot_encoder.get_feature_names_out()))
column_names
```

```python id="d3801242-6f64-4a33-80c7-4155936b4795"
# Convert transformed data into pandas dataframe
X_train_transformed = pd.DataFrame(X_train_transformed, columns=column_names)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=column_names)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="10c2297c-8c9e-42d2-a715-94005c20a524" outputId="e7333907-9f54-4aac-9a90-b526633b16f2"
# Print the transformed data
X_train_transformed
```

```python colab={"base_uri": "https://localhost:8080/"} id="5f36b93a-2b39-444e-9294-e217acee8bec" outputId="e3950e72-a145-454b-9507-e9302f9bb60d"
# Print the information of the transformed training data
X_train_transformed.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 363} id="3627419d-67ba-485a-b5f3-ea4474d4b41b" outputId="ab4bfec8-28e2-4409-8386-807ef05f7327"
X_test_transformed.head(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="a73ba1c6-0e98-426e-82e5-0a529fa58836" outputId="354b0230-5408-425a-8298-e3c189eed9f4"
# Print the information of the transformed testing data
X_test_transformed.info()
```

<!-- #region id="3acd212e-c256-4508-a2a2-ab45ff822808" -->
## **Preprocessing with `ColumnTransformer`**
<!-- #endregion -->

```python id="08d4ed97-54cb-4598-84f6-953be18e6ff8"
# Create an object of the ColumnTransformer
transformer = ColumnTransformer(transformers=[
    ("tranformer_1", SimpleImputer(strategy='mean'), ["fever"]),
    ("transformer_2", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
    ("transformer_3", OneHotEncoder(drop="first", sparse_output=False), ["gender", "city"])
], remainder="passthrough")
```

```python colab={"base_uri": "https://localhost:8080/"} id="184d6720-0d29-4f25-8b80-c3d781c56d64" outputId="f900c279-7480-47ac-c3af-45a48fba22ba"
# Fit and transform the training data
X_train_transformed = transformer.fit_transform(X_train)
X_train_transformed.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="3f70033f-bb03-42c1-8c5c-9fa58a242bd2" outputId="15664858-6ee5-4182-8c70-dfb465c53ee8"
# Transform the testing data
X_test_transformed = transformer.transform(X_test)
X_test_transformed.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="dde17647-0287-4823-98ea-7db0360d29ec" outputId="2e38af5a-42ba-44dc-cf1a-5be5826c064a"
# Checking the new column names of the transformed data
transformer.get_feature_names_out()
```

```python id="24bec896-474e-45d7-bd75-2a27b0f1f6e7"
# Convert the transformed array into pandas dataframe
X_train_transformed = pd.DataFrame(X_train_transformed, columns=transformer.get_feature_names_out())
X_test_transformed = pd.DataFrame(X_test_transformed, columns=transformer.get_feature_names_out())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 443} id="da00b5ca-cf7d-4fbd-b00d-59117e153503" outputId="43397990-9efa-465a-f063-0e22613131bd"
X_train_transformed
```

```python colab={"base_uri": "https://localhost:8080/", "height": 383} id="6daba95a-d809-4653-b743-c7124b89af3a" outputId="8b7f163a-ac27-433e-f79c-d2cb4ed985a4"
X_test_transformed.head(10)
```

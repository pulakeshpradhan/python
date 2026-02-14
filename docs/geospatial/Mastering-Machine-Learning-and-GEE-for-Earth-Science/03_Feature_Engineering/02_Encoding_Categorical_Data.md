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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/03_Feature_Engineering/02_Encoding_Categorical_Data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="1b53db48-f4b4-4efc-8878-c808042a18a3" -->
# **Encoding Categorical Data**
Encoding categorical data is an essential step in preparing data for machine learning models since most machine learning algorithms require numerical input data. Categorical data represents non-numeric data such as categories, labels, or classes.

In Python, you can use various techniques to encode categorical data, and the choice of encoding method depends on the nature of your data and the machine learning algorithm you plan to use.
<!-- #endregion -->

<!-- #region id="6798857e-90ac-44d7-bebb-73aa66dc6e2f" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7q6G4UrQXyft" outputId="2e6e164c-3199-4dc5-efd2-8e70b2469090"
from google.colab import drive
drive.mount("/content/drive/")
```

```python id="9b663240-d39a-4652-a6db-e339c66c790e"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
```

<!-- #region id="133c1e05-93e8-4f5d-9381-acf26b625223" -->
## **Read the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="4c0911b0-638c-41e8-ac84-ec1eb31653c1" outputId="98d3d3cd-d0fa-44be-a917-7e316952a862"
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/customer.csv")
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="7279d5bc-1143-46ed-8202-31e1aae8a05a" outputId="d1f05fea-1eb4-4db6-9d46-6289b957c40d"
df.shape
```

```python id="c0442293-408a-4647-9233-f37e9ef4015d"
# Extrcting the 'review', 'education' and 'purchased' colums from the dataframe
df = df.iloc[:, 2:]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="7a598d4d-0988-4b21-82ad-9a5b193a823a" outputId="b22a1f77-b9d6-4949-ae36-3bd3fbd2d377"
df.head()
```

<!-- #region id="51d4219b-fbc0-4b6a-89dc-f42813c42442" -->
## **Train Test Split**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="c2ffb602-1945-4019-9b35-6f7f74330029" outputId="2f0c4d23-3735-4016-9f4d-a780da205b88"
X_train, X_test, y_train, y_test = train_test_split(df.drop("purchased", axis=1),
                                                    df["purchased"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

<!-- #region id="3b728b3e-63a3-4e58-9f6f-e61a7ecaeb9f" -->
## **Ordinal Encoding**
Ordinal encoding is a technique for encoding categorical data where the categories have a meaningful order or ranking. This method assigns a unique integer value to each category based on its order or priority. Ordinal encoding is appropriate when the categorical data represents ordered or ranked values, such as "low," "medium," and "high" or "small," "medium," "large."
<!-- #endregion -->

<!-- #region id="T7vU07eKZG97" -->
<center><img src="https://miro.medium.com/v2/resize:fit:654/1*NUzgzszTdpLPZpeKPPf0kQ.png" width="40%"></center>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GtGQLmodZn2R" outputId="c45e4e59-c7c4-4b63-ed26-7ea3da8433f0"
# Checking the unique values in each column
print("Unique values in each column:")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")
```

```python id="c4354c9b-627b-4a8f-8b83-272c7fa33200"
# Creating an object of ordinal encoder class
ordinal_encoder = OrdinalEncoder(categories=[["Poor", "Average", "Good"], ["School", "UG", "PG"]],
                                 dtype=np.int8)
# Fit the training data
ordinal_encoder.fit(X_train)

# Transform the training and testing data
X_train_encoded = ordinal_encoder.transform(X_train)
X_test_encoded = ordinal_encoder.transform(X_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="de9f248e-da76-48ed-af6d-e700173f563e" outputId="c0fe784a-d126-4e62-9680-05a38ac638f8"
ordinal_encoder.categories_
```

```python id="92c263a9-674e-4f3f-a452-8d4d2cbd8a2f"
# Converting the encoded array into pandas dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns=["review", "education"])
X_test_encoded = pd.DataFrame(X_test_encoded, columns=["review", "eucation"])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="75ab867f-a3f6-4bbe-9af7-b948f2df3729" outputId="b47e511c-bd46-4a18-90f8-461a72b0e1e4"
# Print the non-encoded training data
X_train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="27495f5c-4ebb-4322-81d5-9f12f66cccc0" outputId="a7bc83b6-4b65-495e-addf-94b9e1d0b10c"
# Print the encoded training data
X_train_encoded.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="0ef4bab4-ae34-4035-a73d-cf5c2b358ac8" outputId="eea2fc6f-ad3d-4a80-db58-7091520b1c74"
# Print the encoded testing data
X_test_encoded.head()
```

<!-- #region id="4b51282d-fe51-43a6-9f87-c80ee5029d0e" -->
## **Label Encoding**
Label encoding is a technique for encoding categorical data into numerical values, where each category is assigned a unique integer label. This encoding is suitable for categorical data where there is no inherent order or ranking among the categories.

You can use the `LabelEncoder` class from the sklearn.preprocessing module to perform label encoding. This encode target labels with value between 0 and n_classes-1. This transformer should be used to encode target values, i.e. y and not the input X.
<!-- #endregion -->

<!-- #region id="y1n_tNmId50B" -->
<center><img src="http://ai-ml-analytics.com/wp-content/uploads/2020/08/encoding-3.png"></center>
<!-- #endregion -->

```python id="f4e0c0e3-2914-4c62-9ee6-2f9bdd136577"
# Creating an object of the label encoder class
label_encoder = LabelEncoder()

# Fit the training data
label_encoder.fit(y_train)

# Transform the training and testing data
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="8a3f0535-f4cd-4459-8972-1432ad01f1b2" outputId="2c6b4a59-5c3a-477d-c0bc-907c80406b4a"
label_encoder.classes_
```

```python colab={"base_uri": "https://localhost:8080/"} id="e05df3bf-0702-4a0f-92c2-7edf97dc595c" outputId="bb996a70-950e-46d1-c8d2-240bf5e80879"
# Print the y_train data
y_train.head(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="88a79a59-c49f-44b9-a376-583f2988e617" outputId="3f42f60e-a6d1-4d22-8e35-48e523e57266"
# Print the y_train_encoded data
y_train_encoded
```

```python colab={"base_uri": "https://localhost:8080/"} id="c179ac7c-6114-4800-b092-7c3a098a37fc" outputId="862eae9e-6cf6-48ba-bfbc-44e4a4e99b09"
# Print the y_test_encoded data
y_test_encoded
```

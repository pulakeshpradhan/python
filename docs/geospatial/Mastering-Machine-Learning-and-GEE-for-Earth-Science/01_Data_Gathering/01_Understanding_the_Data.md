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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/01_Data_Gathering/01_Understanding_the_Data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="6111d659-c603-42c0-8fb5-ed8911a2cdfe" -->
# **Understanding the Data**
Understanding the data is a critical step in the data science process. It involves gaining insights into the structure, content, quality, and characteristics of the data you're working with. Properly understanding the data sets the foundation for making informed decisions, building accurate models, and deriving meaningful insights.
<!-- #endregion -->

<!-- #region id="00233349-36fa-4768-9768-b753d30af2e9" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PawDZoMwpXnE" outputId="1e3a0968-9f4c-4cef-9cd4-7ea488ab1dd5"
from google.colab import drive
drive.mount('/content/drive')
```

```python id="0729fb1c-c7ed-4699-8a41-f60b407862c0"
import pandas as pd
```

<!-- #region id="06847b7b-7d43-47f1-aa7c-30a24aae559d" -->
## **Read the Data**
<!-- #endregion -->

```python id="cab24e9c-40aa-4bbc-ad8c-43d1e89b9e45"
df = pd.read_csv(r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/Global YouTube Statistics.csv", encoding="latin")
```

<!-- #region id="eacd944c-2f70-427c-9a97-0dd3bf73b29b" -->
## **How Big is the Data?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4cfdbea8-2ca8-4730-8779-28de98befb57" outputId="828f9d22-111a-444e-bdc2-0a449b051ce1"
df.shape
```

<!-- #region id="243cb175-a59d-497e-b647-f93a384815e3" -->
## **How does the Data look like?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="933de3d1-318d-4155-b327-116664f8fa6c" outputId="166bf67d-06c1-4fc8-c5d0-adfffdefffee"
# Print first 5 rows of the dataframe
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 429} id="18828cf0-83a9-44e6-a020-6b690d1882fe" outputId="83e3ddd6-40b8-4425-9f28-59364fd697d2"
# Randomly choose 5 rows and print it
df.sample(5)
```

<!-- #region id="9c20e644-a4a8-4fe1-8bdb-6665211c7992" -->
## **What are the Data Types of the Columns?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1c963531-1111-492a-a377-e5333c6d0f92" outputId="a8c928b1-c92e-44c4-bc4a-2469e59f81f2"
df.info()
```

<!-- #region id="46dcd22b-4b66-4490-8251-d6241c610625" -->
## **Are there any Missing Values?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9d5e693a-5a4f-44c0-a9d9-7937c472c654" outputId="acc64ed2-bee9-48fc-86cf-d8db1d2e837d"
# Checking the number of missing values for each column
df.isnull().sum()
```

<!-- #region id="71648034-b5a1-403f-aa05-d1e79e125ee3" -->
## **How does the Data look Mathematically?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="06b175e1-fae3-4441-b6b1-4f9a57987bc7" outputId="d238093a-e7d3-4773-d911-c3d23dad8fae"
df.describe()
```

<!-- #region id="25783af0-db46-4863-880d-63dbbef1cc2a" -->
## **Are there any Duplicate Rows?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="554b97e7-8d80-4e36-ba46-836c8ea891d9" outputId="91d651a4-8bb8-4123-f95e-2c09cf126e6f"
df.duplicated().sum()
```

<!-- #region id="97a20184-dbd2-408d-a782-13aaa0d5ab11" -->
## **How is the Correlation between Columns?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 844} id="7d3810d1-c41c-40da-8f9d-ae1a84bbf91c" outputId="bb37e174-d589-4a41-b38d-adba9c855f68"
# Extract the correlation between all the variables
df.corr()
```

```python colab={"base_uri": "https://localhost:8080/"} id="97318239-dc13-4b34-99ad-bf80ec0fd148" outputId="237931f3-99c3-472f-d2a6-b2c5da883da8"
# Extract the correlation between the 'subscribers' and other numerical columns
df.corr()["subscribers"]
```

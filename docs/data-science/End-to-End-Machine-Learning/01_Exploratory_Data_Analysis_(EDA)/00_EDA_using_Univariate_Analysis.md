[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/01_Exploratory_Data_Analysis_(EDA)/00_EDA_using_Univariate_Analysis.ipynb)

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

# **EDA Using Univariate Analysis**
Exploratory Data Analysis (EDA) using univariate analysis focuses on examining individual variables in your dataset to gain insights into their distribution, characteristics, and potential outliers. Univariate analysis helps you understand the basic properties of each variable and identify patterns that can guide further analysis.


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

## **Read the Data**

```python
df = pd.read_csv(r"D:\Coding\Datasets\titanic.csv")
df.head()
```

## **Categorical Data**


### **Count Plot**

```python
df.shape
```

```python
# Checking the number of people died and survived
sns.countplot(x=df["Survived"])
df["Survived"].value_counts()
```

```python
# Checking the number of people boared at each class
sns.countplot(x=df["Pclass"])
```

```python
# Checking the number of male and female
sns.countplot(x=df["Sex"])
```

```python
# Checking the travellers embarked location
sns.countplot(x=df["Embarked"])
```

### **Pie Chart**

```python
# Plot  the percent of people died and survived in a pie chart
df["Survived"].value_counts().plot(kind="pie", autopct="%.2f")
```

## **Numerical Data**


### **Histogram**

```python
sns.histplot(x=df["Age"], bins=40)
```

### **Dist Plot/KDE Plot**

```python
sns.kdeplot(data=df["Age"])
```

### **Box Plot**

```python
sns.boxplot(x=df["Age"])
```

## **Descriptive Statistics**

```python
# Checking the minimum 'Age'
df["Age"].min()
```

```python
# Checking the maximum 'Age'
df["Age"].max()
```

```python
# Checking the mean 'Age'
df["Age"].mean()
```

```python
# Checking the skewness of the 'Age' column
df["Age"].skew()
```

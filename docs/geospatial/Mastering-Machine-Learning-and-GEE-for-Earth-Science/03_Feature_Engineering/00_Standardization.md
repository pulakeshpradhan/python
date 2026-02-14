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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/03_Feature_Engineering/00_Standardization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="615dce42-564e-42c7-aac3-d3e5a6c4f068" -->
# **Standardization - Feature Scaling**
Standardization, also known as z-score normalization or standard scaling, is a technique used in data preprocessing to rescale the features of a dataset. The goal of standardization is to transform the data so that it has a mean of 0 and a standard deviation of 1. This process helps to make the features more comparable and can be particularly useful in machine learning algorithms that are sensitive to the scale of the input features, such as gradient descent-based optimization algorithms.
<!-- #endregion -->

<!-- #region id="c7f69f1e-071e-4cb8-b985-1f7495848e97" -->
<center><img src="https://miro.medium.com/v2/resize:fit:552/1*DK6tNx7Ke_27-CdLT3_1Ug.png"></center>
<!-- #endregion -->

<!-- #region id="57872421-98f0-4712-b58c-1d1e42e8f986" -->
**1. Standard Deviation:** <br>
The standard deviation ($\sigma$) is a measure of the amount of variation or dispersion in a set of values. The formula for calculating the standard deviation of a sample is different from the formula for calculating the standard deviation of a population.

For a `Sample`:

$$s = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}}$$

where:
- $s$ is the sample standard deviation,
- $n$ is the number of observations in the sample,
- $x_i$ is each individual observation in the sample,
- $\bar{x}$ is the sample mean.

For a `Population`:

$$\sigma = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}}$$

where:
- $\sigma$ is the population standard deviation,
- $N$ is the number of observations in the population,
- $x_i$ is each individual observation in the population,
- $\mu$ is the population mean.

<br>

**2. Z-score:**<br>
A Z-score, also known as a standard score, is a statistical measure that quantifies how far away a particular data point is from the mean (average) of a dataset when measured in terms of standard deviations. It's a way to standardize and normalize data, making it easier to compare values from different datasets or different parts of the same dataset.

$$Z = \frac{(X - \mu)}{\sigma}$$

where:
- $Z$ is the standardized value,
- $X$ is the original value of the feature,
- $\mu$ is the mean of the feature,
- $\sigma$ is the standard deviation of the feature.
<!-- #endregion -->

<!-- #region id="cc6da705-be7d-496b-916c-919473b4f794" -->
## **When to use Standardization?**
<!-- #endregion -->

<!-- #region id="10d8055f-039d-45b5-8daa-bccb2b12a4b8" -->
<center><img src="https://miro.medium.com/max/1400/1*qRmiffZgkNaXnTBZwDafCA.png" width="80%"></center>
<!-- #endregion -->

<!-- #region id="a1a290b2-49e1-4b12-abdc-905cdde2849b" -->
# **Example**
<!-- #endregion -->

<!-- #region id="597b7711-8761-416f-8bd0-36c8516feba9" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SoywawFvcrtf" outputId="63bada6e-632e-42b8-ef07-b2c7d614fcf7"
from google.colab import drive
drive.mount('/content/drive')
```

```python id="7c25c2b2-736c-4ef3-a2c1-fc0841bd0b77"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```

<!-- #region id="082487f4-7326-423a-bafd-95573a3b633d" -->
## **Read the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="5b70dfb3-487f-4872-82fd-17cfdbf8837d" outputId="91f1be78-33eb-46c8-a785-e729b4ed6652"
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/Social_Network_Ads.csv")
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="68c64ef5-73c6-484f-ba0f-e07353e7b167" outputId="3d021044-1463-4288-ed38-69f083e27c9f"
df.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="37651496-bee0-49f7-972d-4903a01e4a8b" outputId="22070045-923e-45c1-d017-9db93a90dba2"
# Check the null values
df.isnull().sum()
```

<!-- #region id="84547301-2fcf-4d84-83b2-600d666d9945" -->
## **Train Test Split**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="26eb9133-8fb8-4162-8f5d-30d4abc5d0ef" outputId="814cb56a-dabe-4c11-a638-5d1344e21b43"
# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(df.drop("Purchased", axis=1),
                                                    df["Purchased"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

<!-- #region id="f684fde0-3c63-416c-882c-e3e2ffe74030" -->
## **Standard Scaler**
In scikit-learn, the `StandardScaler` is a preprocessing technique provided by the library for standardizing or scaling features in your dataset. It follows the standardization process discussed earlier, where it scales the features to have a mean of 0 and a standard deviation of 1. This is done to ensure that all features have the same scale, making them more suitable for machine learning algorithms that are sensitive to feature scales.

The `fit` method is typically called on a machine learning model or a data preprocessing object to adapt it to the specific dataset you are working with. Its purpose is to learn from the data and update the internal state of the object.
<!-- #endregion -->

```python id="afd045be-dc46-4c4f-994e-3ebfa5c17877"
# Instantiate a 'StandardScaler' object
scaler = StandardScaler()

# Fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# Transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python id="09667611-c74f-4d93-a48f-277ca9ef43ed"
# transform always returns a numpy array
# Converting the scaled numpy arrays into pandas dataframes
X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(data=X_test_scaled, columns=X_test.columns)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="50920ad7-6e6e-4911-b9c4-6955fc75c0e9" outputId="e66d6328-ccd3-4270-cf06-348bd2a0b939"
X_train_scaled.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="38511e54-d48f-444f-9539-5dfc453438d4" outputId="af0e40e3-36f8-4049-c919-fff466dac0a5"
# Describe the training data
X_train.describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="6ad06ab1-2bd3-455b-8d0c-839b1334acad" outputId="3d323b7d-3087-4021-e631-4cad9510fb68"
# Describe the scaled training data
X_train_scaled.describe().round(2)
```

<!-- #region id="2cfbddf0-474b-4225-a48d-af5314343669" -->
## **Effect of Scaling**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 486} id="31edeac9-64b4-4e8c-a67d-30b122036edd" outputId="403719d5-30f5-4b1d-931c-77bcd18bdb94"
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a scatter plot of the training data
ax1.scatter(x=X_train["Age"], y=X_train["EstimatedSalary"])
ax1.set_title("Before Standardization")
ax1.set_xlabel("Age")
ax1.set_ylabel("Estimated Salary")

# Creating a scatter plot of the scaled training data
ax2.scatter(x=X_train_scaled["Age"], y=X_train_scaled["EstimatedSalary"], color="red")
ax2.set_title("After Standardization")
ax2.set_xlabel("Standardized Age")
ax2.set_ylabel("Standardized Estimated Salary")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 487} id="aeebc391-ca9f-48d5-aa89-2d3f6f80b7aa" outputId="bd0a912d-cb5d-47f8-c450-7bcf39abb7f8"
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a probabilty density plot of the training data
ax1.set_title("Before Standardization")
sns.kdeplot(data=X_train["Age"], ax=ax1, label="Age")
sns.kdeplot(data=X_train["EstimatedSalary"], ax=ax1, label="Estimated Salary")
ax1.legend()

# Creating a probabilty density plot of the scaled training data
ax2.set_title("After Standardization")
sns.kdeplot(data=X_train_scaled["Age"], ax=ax2, label="Age")
sns.kdeplot(data=X_train_scaled["EstimatedSalary"], ax=ax2, label="Estimated Salary")
ax2.legend()

plt.show()
```

<!-- #region id="e21c2343-3d3e-416a-adbb-5a44fa8ce4a3" -->
## **Comparison of Distribution**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 487} id="8b6a9532-34b9-4e3d-96f3-62a5db9e49b6" outputId="877fedbd-ce57-414f-9150-c9e7ac72d0aa"
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Creating a probabilty density plot of the 'Age' column from the training data
sns.kdeplot(X_train["Age"], ax=ax1)
ax1.set_title("Age Distribution before Scaling")

# Creating a probabilty density plot of the 'Age' column from the scaled training data
sns.kdeplot(X_train_scaled['Age'], ax=ax2)
ax2.set_title("Age Distribution after Scaling")

plt.show()
```

<!-- #region id="eb1fe934-a198-4536-b2e3-1591ca4d42e6" -->
## **Why Scaling is Important?**
<!-- #endregion -->

<!-- #region id="7c559b1a-7b88-4a43-857a-34b9495ef53b" -->
### **Comparison on Logistic Regression Model**
<!-- #endregion -->

```python id="eed5a239-3803-48e9-af0c-f40ed7da3d54"
# Creating two logistic regression model for the training data and scaled training data respectively
lr = LogisticRegression()
lr_scaled = LogisticRegression()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 74} id="bc41bf94-d582-4707-9a33-bfe627b89a70" outputId="825eb414-7ccd-40e2-8eb6-5a534fc1aae9"
# Fitting the data to the models
lr.fit(X=X_train, y=y_train)
lr_scaled.fit(X=X_train_scaled, y=y_train)
```

```python id="0bcccd8a-a3c8-4f18-9372-eafdc05e5a3a"
# Predict the testing data
y_pred = lr.predict(X_test)
y_pred_scaled = lr_scaled.predict(X_test_scaled)
```

```python colab={"base_uri": "https://localhost:8080/"} id="cbe756fa-d4b3-4c52-9ee1-3fead7878310" outputId="b1826d3e-6bfe-4283-d8d4-b756280c5fbb"
# Calculating the accuracy
print("Accuracy Score on Actual Data:", accuracy_score(y_test, y_pred).round(2))
print("Accuracy Score on Scaled Data:", accuracy_score(y_test, y_pred_scaled).round(2))
```

<!-- #region id="22518ed0-f5a7-4db3-8431-2ef39f438c1d" -->
### **Comparison on Decision Tree Model**
<!-- #endregion -->

```python id="a74c6dbc-3a78-4a85-9091-78d9a080d3a7"
# Creating two decision tree model for the training data and scaled training data respectively
dt = DecisionTreeClassifier()
dt_scaled = DecisionTreeClassifier()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 74} id="c5460e78-4aec-498f-9362-9d6a471ebea6" outputId="f56554d0-e0e7-4994-d313-d2b1555bc3cd"
# Fitting the data to the models
dt.fit(X=X_train, y=y_train)
dt_scaled.fit(X=X_train_scaled, y=y_train)
```

```python id="3d736bd1-52f8-44d3-99b2-bf8afcb14beb"
# Predict the testing data
y_pred = dt.predict(X_test)
y_pred_scaled = dt_scaled.predict(X_test_scaled)
```

```python colab={"base_uri": "https://localhost:8080/"} id="14399866-ecf6-48fa-96fb-fc332332cc4f" outputId="53809d6a-94a1-45e7-bea5-44fef19b7151"
# Calculating the accuracy
print("Accuracy Score on Actual Data:", accuracy_score(y_test, y_pred).round(2))
print("Accuracy Score on Scaled Data:", accuracy_score(y_test, y_pred_scaled).round(2))
```

<!-- #region id="lZY0YTyCg9fz" -->
🤔 **Note:**<br>
The accuracy improved after applying normalization to the Logistic Regression model because Logistic Regression is sensitive to the scale of input features, and normalization ensures a consistent and effective learning process by bringing all features to a similar scale. On the other hand, the Decision Tree model's accuracy remained unchanged because Decision Trees are inherently less sensitive to the scale of input features, as their splitting criteria depend on feature thresholds rather than absolute values. Therefore, normalization did not significantly impact the Decision Tree model's performance.
<!-- #endregion -->

<!-- #region id="7fd9ff30-08bc-4b9f-a655-f32fbfe6d9e6" -->
## **Effect of Outlier**
Standardization, by itself, does not handle outliers in the data. In fact, standardization can sometimes exacerbate the impact of outliers, making them more prominent in the scaled data. Standardization can be affected by outliers, and its sensitivity to extreme values may impact the resulting standardized values. Depending on the context and goals of the analysis, alternative scaling methods that are more robust to outliers might be considered.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 143} id="c89b5dd4-0b1c-4861-ba1d-a4bcce3bface" outputId="768d41da-ed94-461b-98d9-adf476a7fc3a"
# Adding some outliers to the data
outliers = pd.DataFrame({"Age":[10, 90, 97], "EstimatedSalary":[1000, 250000, 350000], "Purchased": [0, 1, 1]})
outliers
```

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="d9487bd4-68d4-4304-b9ac-7acde6efd6e3" outputId="f6043ef6-85c7-4d7d-efbf-52f5a65e97d3"
# Concat the outliers in the previous dataframe
new_df = pd.concat([df, outliers])
new_df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="2133020a-9076-4dc4-9f0b-b72249890f68" outputId="c3411229-caab-4e29-b4a3-8db1e5122cec"
# Create a scatter plot
plt.scatter(x=new_df["Age"], y=new_df["EstimatedSalary"])
plt.show()
```

<!-- #region id="d1d0133a-a59e-4803-9675-d7e59efd5e89" -->
### **Applying Standardization**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cdebac6b-e52d-4d05-9e84-acef2968ec32" outputId="dcc948a8-ed4f-466e-8ee2-7aba167ba22a"
# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(new_df.drop(["Purchased"], axis=1),
                                                    new_df["Purchased"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```

```python id="f4179cef-201a-4d3e-99fe-bdfcb33fcc30"
# Creating a Standard Scaler object
scaler = StandardScaler()

# Fitting the data
scaler.fit(X_train)

# Applying Standardization on the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="9922875d-06bb-40aa-a042-70d5b48dacf9" outputId="d0f53390-65a7-4f89-939b-e2da55ec0a1a"
# Converting the scaled data into a pandas dataframe
X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(data=X_test_scaled, columns=X_test.columns)
X_train_scaled.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 503} id="622d9bb8-b3df-4b8a-916f-cf5fec5ee84b" outputId="f3152c67-29cb-472a-9202-cc4487636aa0"
# Plot the data
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Plotting the x_train data
ax1.scatter(x=X_train["Age"], y=X_train["EstimatedSalary"])
ax1.set_title("Before Standardization")
ax1.set_xlabel("Age")
ax1.set_ylabel("Estimated Salary")

# Plotting the scaled x_train data
ax2.scatter(x=X_train_scaled["Age"], y=X_train_scaled["EstimatedSalary"], color="red")
ax2.set_title("After Standardization")
ax2.set_xlabel("Standardized Age")
ax2.set_ylabel("Standardized Estimated Salary")
```

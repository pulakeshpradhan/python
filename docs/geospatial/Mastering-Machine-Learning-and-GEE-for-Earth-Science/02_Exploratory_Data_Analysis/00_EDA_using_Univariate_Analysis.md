[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Mastering-Machine-Learning-and-GEE-for-Earth-Science/02_Exploratory_Data_Analysis/00_EDA_using_Univariate_Analysis.ipynb)

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

<!-- #region id="90a4c25d-1f41-4819-88dc-087dce12ee98" -->
# **EDA Using Univariate Analysis**
Exploratory Data Analysis (EDA) using univariate analysis focuses on examining individual variables in your dataset to gain insights into their distribution, characteristics, and potential outliers. Univariate analysis helps you understand the basic properties of each variable and identify patterns that can guide further analysis.
<!-- #endregion -->

<!-- #region id="2c33f5c7-ded0-4fda-a740-b042d4ddca04" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mNqCUtBj3toW" outputId="a36c99b9-2241-40bd-9e9d-3c5534e0a12d"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="d60ea2b2-da57-4889-9fb3-03b07578e068"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

<!-- #region id="9f83df03-5c96-448f-a2f1-5bec3a4bb351" -->
## **Read the Data**

#### **Dataset Description**
The Titanic dataset is a well-known and frequently used dataset in the field of machine learning. It contains information about passengers who were on board the RMS Titanic during its ill-fated maiden voyage, which tragically ended in a sinking on April 15, 1912. This dataset is often employed for practicing and demonstrating various machine learning techniques, particularly in the context of binary classification.

The dataset typically includes the following features:

1. **PassengerID:** A unique identifier for each passenger.
2. **Survived:** A binary variable indicating whether the passenger survived the sinking (1) or did not survive (0).
3. **Pclass (Ticket Class):** The socio-economic class of the passenger, with values 1, 2, or 3.
4. **Name:** The name of the passenger.
5. **Sex:** The gender of the passenger.
6. **Age:** The age of the passenger. This feature may contain missing values.
7. **SibSp:** The number of siblings/spouses aboard the Titanic.
8. **Parch:** The number of parents/children aboard the Titanic.
9. **Ticket:** The ticket number.
10. **Fare:** The amount of money paid for the ticket.
11. **Cabin:** The cabin number where the passenger stayed. This feature may contain missing values.
12. **Embarked:** The port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

The main objective when working with the Titanic dataset in a machine learning context is often to predict whether a passenger survived or not based on the given features. This is a binary classification problem. Machine learning algorithms can be trained on a subset of the data to learn patterns and relationships between the features and the target variable (Survived), and then these models can be evaluated on unseen data to assess their predictive performance.

It's important to note that the dataset might require preprocessing, such as handling missing values, encoding categorical variables, and scaling numerical features, to prepare it for machine learning models. The Titanic dataset serves as a valuable resource for beginners to practice and showcase their machine learning skills.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 275} id="ce3f9e36-f504-450e-8546-f3571bca146b" outputId="30ab3e7c-8dfa-40cf-c887-a621c95b757e"
# Read the data into a dataframe
df = pd.read_csv(r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/titanic.csv")
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="w7UDtvyo579l" outputId="c77031d0-840a-4e6e-d6e6-3726e6c0bd70"
# Print the shape of the data
df.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="h6XmRppa54Qs" outputId="512332a2-1d4c-40cf-f8d9-de54c4adb404"
# Check the information of all the columns
df.info()
```

<!-- #region id="8a3105a8-e2ba-4052-b507-40a081e09090" -->
## **EDA on Categorical Data**
Exploratory Data Analysis (EDA) on categorical data involves analyzing and visualizing the patterns, relationships, and distributions within categorical variables. Here are some common techniques and tools used for EDA on categorical data:
<!-- #endregion -->

<!-- #region id="4c07b3d6-e57a-4d6f-84bd-6ffbd4096fec" -->
### **Count Plot**

A count plot is a type of categorical data visualization that displays the counts of observations in each category of a categorical variable. It is particularly useful for understanding the distribution and frequency of different categories within a dataset.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 503} id="wVrKZoOi7vk6" outputId="00f41bc5-1f79-4956-c100-c4f3d69a3554"
# Check the number of male and female in the 'Sex' column
sns.countplot(x=df["Sex"])
df["Sex"].value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 501} id="0f96ca45-bea3-4621-9765-f3035e421b2c" outputId="6eed1db2-5605-4e1b-b348-58857a5555b1"
# Check the number of people died and survived
sns.countplot(x=df["Survived"])
df["Survived"].value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 518} id="b2420234-ac50-4147-b537-8bb2a8b8f7e8" outputId="c486e2b8-7d16-459e-9df5-7f985e9a911c"
# Check the number of people boared at each class
sns.countplot(x=df["Pclass"])
df["Pclass"].value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 518} id="d6f4829c-aa21-478c-ae9b-efca2d5da57c" outputId="e6643997-424d-49f9-9931-163994c3e9c0"
# Check the travellers embarked location
sns.countplot(x=df["Embarked"])
df["Embarked"].value_counts()
```

<!-- #region id="3690778d-3b24-4649-97f9-64b863119fa9" -->
### **Pie Chart**
A pie chart is a circular statistical graphic that is divided into slices to illustrate numerical proportions. It is a popular visualization tool for representing the distribution of a categorical variable as a whole. Each slice of the pie represents a specific category, and the size of each slice corresponds to the proportion or percentage of the whole that the category represents.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 458} id="557bafb2-dc44-42b1-b647-1ebe5c6b2003" outputId="448ff52d-398b-4485-e58f-d12f384e4c95"
# Plot the percent of people died and survived in a pie chart
df["Survived"].value_counts().plot(kind="pie", autopct="%.2f")

# Calculate the percentage
(df["Survived"].value_counts()/len(df["Survived"])) * 100
```

<!-- #region id="858cdd75-8290-4d64-9054-1f55209304d5" -->
## **EDA on Numerical Data**

Exploratory Data Analysis (EDA) on numerical data involves analyzing and visualizing the patterns, distributions, and relationships within quantitative variables. Here are some common techniques and tools used for EDA on numerical data:
<!-- #endregion -->

<!-- #region id="7f558ac7-7291-4fa0-b40a-eedca7a943fc" -->
### **Histogram**
Create histograms to visualize the distribution of numerical variables. Histograms display the frequency or probability density of different ranges or bins.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="64251ba4-5cf0-425e-a79a-1bfe728e4504" outputId="03302c37-db29-4c9e-dff6-a1bd2befd0e4"
# Plot the histogram of the 'Age' column
sns.histplot(x=df["Age"], bins=50)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 468} id="iyKKstT8B5wP" outputId="c1a7c435-3192-4c61-9580-1f89ce52bcb6"
# Plot the histogram of the 'Age' column
sns.histplot(df["Fare"], bins=50)
```

<!-- #region id="675df42a-8306-4cf3-9f7d-da193e9057ff" -->
### **Dist Plot/KDE Plot**
Kernel density plots provide a smooth estimation of the probability density function for a numerical variable, offering insights into the underlying distribution.

Interpreting the Probability Density Function (PDF) of the 'Age' column in the Titanic dataset involves analyzing the smoothed estimate of the likelihood of different age values occurring among the passengers. Here's how you can interpret the PDF of the 'Age' column in the Titanic dataset:

1. **Central Tendency:**
   - Look for the central tendency in the PDF. Peaks in the PDF may indicate modes or clusters of age values that are more prevalent among the passengers.
   - The highest point in the PDF might correspond to the average or median age of the passengers.

2. **Distribution Shape:**
   - Observe the overall shape of the PDF. A symmetric distribution suggests a balanced age distribution, while asymmetry may indicate skewness.
   - Skewness to the right (positive skew) suggests a longer right tail, indicating a higher presence of older passengers. Skewness to the left (negative skew) suggests a higher presence of younger passengers.

3. **Variability and Spread:**
   - Examine the width of the PDF at different points. A wider PDF indicates greater variability or dispersion in age values, while a narrower PDF suggests less variability.

4. **Distinct Age Groups:**
   - Identify if there are distinct peaks or modes in the PDF, which may represent specific age groups.
   - For example, there might be peaks around certain ages corresponding to children, adults, or elderly passengers.

5. **Outliers:**
   - Look for isolated peaks or regions with unusually low probability density, which may indicate the presence of outliers or unique age values.

6. **Comparison with Other Variables:**
   - Compare the 'Age' PDF with other relevant variables (e.g., 'Survived', 'Class', 'Gender') to identify potential relationships or patterns. For instance, you might compare the age distribution between survivors and non-survivors.

7. **Handling Missing Values:**
   - Address any missing values in the 'Age' column, as these may impact the shape and interpretation of the PDF. You can choose to impute missing ages or visualize the distribution excluding missing values.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="f4dc04f0-40c8-4f6f-a242-4f0a67fd5613" outputId="6b317dd2-4fe6-4a46-c11f-c075f23fc007"
# Plot the PDF of the 'Age' column
sns.kdeplot(data=df["Age"])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="2qlZhW3RFPrj" outputId="3901c442-259e-4544-d4b3-3fe42ec7ec1c"
# Plot the PDF of the 'Age' column by the gender
sns.kdeplot(data=df, x="Age", hue="Sex")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 468} id="Q4xYBn_5HCdk" outputId="bd6856ee-314c-48c9-89fc-7e6c002e8377"
# Compare the 'Age' PDF with 'Survived' to identify potential relationships or patterns
sns.kdeplot(data=df, x="Age", hue="Survived", palette=["red", "green"])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="HL-Ha3hOCcNO" outputId="7bc16a2b-11e8-4bb4-ba32-ee8cf4b4a7d3"
# Plot the PDF of the 'Fare' column
sns.kdeplot(data=df["Fare"])
```

<!-- #region id="842fbdcb-2141-4a94-a3c3-e95861e59659" -->
### **Box Plot**
A boxplot, also known as a box-and-whisker plot, is a graphical representation of the distribution of a dataset. It provides a visual summary of the central tendency, spread, and potential outliers within the data. Boxplots are particularly useful for comparing the distribution of numerical variables across different groups or categories.

Here are the key components and interpretations of a boxplot:

1. **Box:**
   - The central box represents the interquartile range (IQR), which spans the middle 50% of the data. The box's bottom and top edges denote the first quartile (Q1) and third quartile (Q3), respectively.
   - The width of the box indicates the spread of the middle 50% of the data.

2. **Median (Q2):**
   - A line inside the box represents the median (Q2), which is the middle value when the data is sorted.

3. **Whiskers:**
   - Whiskers extend from the box to the minimum and maximum values within a specified range. The range can be determined by various methods, such as 1.5 times the IQR or extending to the minimum and maximum values within a certain percentage of the data.
   - Outliers beyond the whiskers are often represented as individual points.

4. **Outliers:**
   - Individual data points outside the whiskers are considered potential outliers. These points are plotted individually, providing a clear visual indication of values that significantly deviate from the central tendency.

5. **Symmetry and Skewness:**
   - The symmetry or skewness of the distribution can be inferred by observing the relative lengths of the whiskers and the position of the median.
   - An asymmetric boxplot with a longer tail on one side suggests skewness.

6. **Comparison Across Groups:**
   - Boxplots are particularly useful for comparing the distribution of numerical variables across different categories or groups. Multiple boxes can be displayed side by side for visual comparison.

7. **Seaborn's `boxplot`:**
   - In Python, the Seaborn library provides the `boxplot` function for creating boxplots.
   - This function offers various customization options, including the ability to create grouped or categorical boxplots.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="218c51f7-e9dd-453a-afe2-ed1eeff39268" outputId="11a64ec0-ee49-43c4-ab67-04fbd8c38601"
# Plot the box plot of 'Age' column
sns.boxplot(x=df["Age"])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="cKkfQktsK8Fj" outputId="66d7bc93-2f55-44e7-ef94-f7b960de7457"
# Plot the 'box' plot of 'Fare' column
sns.boxplot(x=df["Fare"])
```

<!-- #region id="842d7e0f-a599-4b0e-9e48-1d29e24d2d98" -->
## **Descriptive Statistics**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7b4617f2-e56b-4071-9270-94a652525839" outputId="5a9f45fb-0b42-436f-a63f-fe0d3b8a78dd"
# Check the minimum 'Age'
df["Age"].min()
```

```python colab={"base_uri": "https://localhost:8080/"} id="1308f2f2-908d-4a68-a4a5-73d0bb686952" outputId="771674d9-0158-4f73-aa7a-643e3c165589"
# Check the maximum 'Age'
df["Age"].max()
```

```python colab={"base_uri": "https://localhost:8080/"} id="cddc35aa-bd83-421b-a39b-515f87db238c" outputId="e5d48509-9b39-4449-e8ae-9c64f687d914"
# Check the mean 'Age'
df["Age"].mean()
```

```python colab={"base_uri": "https://localhost:8080/"} id="c19a8ed5-9f73-455e-931d-7a823c73eeb1" outputId="dfd4a3be-51b2-4b90-d89d-58f003510464"
# Check the skewness of the 'Age' column
df["Age"].skew()
```

```python colab={"base_uri": "https://localhost:8080/"} id="DXLI6hnDNXCa" outputId="392e80e0-3211-4f3c-9a8d-b5de46ab58ac"
# Check all the descriptive statistics
df["Age"].describe()
```

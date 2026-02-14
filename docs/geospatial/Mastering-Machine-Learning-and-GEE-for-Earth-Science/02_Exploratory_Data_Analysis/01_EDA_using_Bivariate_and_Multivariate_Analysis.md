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

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/02_Exploratory_Data_Analysis/01_EDA_using_Bivariate_and_Multivariate_Analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="f72dfdec-0dea-4f88-b442-4586905b3fb7" -->
# **EDA Using Bivariate and Multivariate Analysis**
Exploratory Data Analysis (EDA) is a critical initial step in the data analysis process that involves summarizing, visualizing, and understanding the main characteristics and patterns present in a dataset. Bivariate and multivariate analysis are two important components of EDA that help you explore relationships and interactions between multiple variables in your dataset.
<!-- #endregion -->

<!-- #region id="1aac4d2a-7a35-4343-9dc5-81eaec402419" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pcxAMNWNgyhr" outputId="30b5b519-acd9-41e1-803c-6944f1e7a96c"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="f75bf1bd-718f-4410-a14e-45f5a480d5db"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="e752081d-8e5c-409e-9e7a-fa81744e956b" -->
## **Read the Data**
<!-- #endregion -->

```python id="5e1048c1-995c-4b4b-98a0-3a2c3fe7932b"
# Reading the 'tips' data from seaborn
tips = sns.load_dataset("tips")
```

```python id="bdd7b6f0-af43-49e0-a4f1-863f74b4c622"
# Read the 'titanic' data
titanic = pd.read_csv(r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/titanic.csv")
```

```python id="fa67470c-84a4-448a-9145-b873ebfeff35"
# Read the 'iris' dataset
iris = pd.read_csv(r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/Iris.csv")
```

```python id="c1795708-8211-4a9d-8902-fb01c5f471bd"
# Read the 'flight' data
flights = pd.read_csv(r"/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/flights.csv")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="ef12d752-7c37-407c-a3f5-d4f819ab296b" outputId="6457270d-3c2c-4f98-a8fb-252672d5fcce"
tips.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="62f0b648-9412-445e-b3d5-797bf715ce88" outputId="6e670ef3-fcb6-4ace-e8a5-63cd70a0e7c1"
tips.shape
```

<!-- #region id="2845dfc7-7cd7-48e9-aa89-c724b0f4d2d1" -->
## **Scatter Plot (Numerical - Numerical)**
A scatterplot is a fundamental and widely used data visualization technique that displays the relationship between two numerical variables. It is particularly effective for revealing patterns, trends, and potential correlations between the variables. In a scatterplot, each data point is represented as a dot, and the position of the dot corresponds to the values of the two numerical variables on the x and y axes.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="522c71fa-b44b-471d-8205-50a0e8c0d862" outputId="d6504594-8e05-4b65-b597-b42979855789"
# Plotting a scatterplot between 'total_bill' and 'tip' using univariate analysis
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 467} id="65513b77-41f7-492b-b05a-97ad0d952ed4" outputId="574dd82d-21c8-4862-c043-80250256ef8f"
# Plotting a scatterplot between 'total_bill' and 'tip' using multivariate analysis
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex", style="smoker", size="size")
plt.legend(bbox_to_anchor=(1, 1))
```

<!-- #region id="eIlwcFiqioXi" -->
🤔 **Note:** <br>

1. **`sns.scatterplot()`**:
   - This function from the Seaborn library is used to create a scatterplot.
   - The `data` parameter specifies the DataFrame (`tips`) containing the data.
   - The `x` and `y` parameters define the variables to be plotted on the x-axis and y-axis, respectively (`total_bill` and `tip`).
   - The `hue` parameter introduces color differentiation based on the values of the "sex" variable.
   - The `style` parameter introduces different marker styles based on the values of the "smoker" variable.
   - The `size` parameter introduces different marker sizes based on the values of the "size" variable.

2. **`plt.legend()`**:
   - This function from Matplotlib is used to display a legend for the scatterplot.
   - The `bbox_to_anchor` parameter specifies the location of the legend box relative to the axes. In this case, it is set to the top-right corner of the plot (`bbox_to_anchor=(1, 1)`).

<!-- #endregion -->

<!-- #region id="283996ba-13c0-4181-9a6b-73fae478956b" -->
## **Bar Plot (Numerical - Categorical)**

A bar plot (or bar chart) is a common and effective data visualization technique used to represent the relationship between a numerical variable and a categorical variable. It is particularly useful for displaying and comparing the distribution of a numerical variable across different categories. In a bar plot, rectangular bars are drawn to represent the values of the numerical variable for each category in the categorical variable.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 258} id="613fcc56-2238-4806-8385-e45c10a1f7f3" outputId="56249efa-0774-4705-ad0b-f956a0853b66"
titanic.head()
```

<!-- #region id="obXCB6lQkqCZ" -->
**Plot Interpretation**:
   - The bar plot aims to show the average age of passengers traveling in each passenger class.
   - Each bar represents a different passenger class (1st, 2nd, or 3rd class).
   - The height of each bar corresponds to the average age of passengers in the respective class.
   - The black lines (error bars) extending from each bar represent one standard deviation from the mean age for each passenger class.
   - The length of the error bars provides a visual indication of the variability or spread of ages within each class.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="ed8ffb5e-8841-4817-97a8-d3f82cf23db2" outputId="395facbb-3a29-41b2-e0b4-d9fcf59fb9fc"
# Plot the avergae age of a passsenger traveling in a particular class
sns.barplot(data=titanic, x="Pclass", y="Age")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="831e748e-11d7-43cc-9141-f6871b096228" outputId="6f37c767-f92d-4055-cc4b-a48cfc0c3f70"
# Plot the avergae age of a passsenger (male and female) traveling in a particular class
sns.barplot(data=titanic, x="Pclass", y="Age", hue="Sex")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="d9e00cab-be15-47a7-9f22-167194cbfa66" outputId="77c70ef1-5bb8-4eaa-fb95-e63fea42ed0c"
# Plot the avergae fare in a particular class
sns.barplot(data=titanic, x="Pclass", y="Fare", hue="Sex")
```

<!-- #region id="fbcbd4f2-daa6-48b2-afb3-50676e932cc9" -->
## **Box Plot (Numerical - Categorical)**

A box plot (or box-and-whisker plot) is a powerful and widely used data visualization tool for representing the distribution of a numerical variable across different categories of a categorical variable. Box plots provide insights into the central tendency, spread, and potential outliers within each category, making them especially valuable for comparing the distributions of numerical data across different groups.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="43692ed3-71c0-4fda-bbf6-5b1468f85818" outputId="4e364a23-f418-4a69-a61e-bf7ba2dce552"
# Plot a boxplot between 'Sex' and 'Age'
sns.boxplot(data=titanic, x="Sex", y="Age")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="e582cd5d-6c6a-4177-8f74-a59b7f4d21e7" outputId="56eec88f-0582-4b89-86cf-bebb9fcffb69"
# Plot a boxplot between 'Sex' and 'Age' also show 'Survived'
sns.boxplot(data=titanic, x="Sex", y="Age", hue="Survived")
```

<!-- #region id="97f342ac-0e27-41dd-8a52-b2af4f18b9ba" -->
## **KDE Plot (Numerical - Categorical)**

A KDE (Kernel Density Estimate) plot is a data visualization technique that provides an estimation of the probability density function of a continuous random variable. In the context of a numerical-categorical relationship, a KDE plot can be used to visualize the distribution of a numerical variable within different categories of a categorical variable. Seaborn's `kdeplot` function is commonly used to create KDE plots in Python.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="3dfc90b0-4629-43f1-afa7-69abca976cef" outputId="ac1df408-0a52-41c6-dc94-0b32e0668621"
# Plot the relationship between 'Age' and 'Survived'
# Plotting the Age of the people who could not survive
sns.kdeplot(titanic[titanic["Survived"]==0]["Age"])
# Plotting the Age of the people who could survive
sns.kdeplot(titanic[titanic["Survived"]==1]["Age"])
```

<!-- #region id="a70406f7-9ba6-42b5-ac01-1beb6c8f6ae5" -->
## **Heat Map (Categorical - Categorical)**
A heatmap is a data visualization technique used to represent the magnitude of a variable in a two-dimensional space, typically through color intensity. In the context of categorical-categorical relationships, a heatmap is particularly useful for visualizing the frequency, proportions, or relationships between two categorical variables. Seaborn's `heatmap` function in Python is commonly used for creating categorical heatmaps.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 258} id="b8bd3533-9a06-4521-a1cc-9e1338ebef4e" outputId="c19b7de5-503d-43aa-fbb9-4377ac3419fb"
titanic.head()
```

<!-- #region id="-K6csjUGpumY" -->
🤔 **Note:** `crosstab` is a function in the Pandas library in Python that computes a cross-tabulation (also known as contingency table) of two or more factors. It is particularly useful for analyzing the relationships between categorical variables by summarizing the frequency of occurrences of different combinations of categories.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 174} id="5bbb9ecd-2d17-45d0-90ba-e6b8ce4e97a9" outputId="b8f14136-d29c-4811-bbea-9645fe7d9538"
# Get the number of people died and survived in each Pclass
pd.crosstab(index=titanic["Pclass"], columns=titanic["Survived"])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="912ca60f-71e3-441e-96e7-e34101c089a0" outputId="b8549f61-9b37-41f8-c0b2-ef7d70ee77cb"
sns.heatmap(pd.crosstab(index=titanic["Pclass"], columns=titanic["Survived"]), cmap="Reds",
            annot=True, fmt="")
```

<!-- #region id="c7lUViqGqqdu" -->
## **Group By Function**

The `groupby` function in Pandas is a powerful tool for splitting a DataFrame into groups based on one or more criteria, applying a function to each group independently, and then combining the results into a new DataFrame. It is a key feature for data analysis and manipulation in Pandas.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 174} id="8eaf7930-27b3-4b02-87a5-254aa46f2f5d" outputId="d6b670ed-e970-4d04-94fa-eef7a1586fa7"
# Group titanic dataframe based on 'Pclass'
titanic.groupby(by="Pclass").mean()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 461} id="507ca8a7-9c5d-4fa5-9655-2b06b5d53d08" outputId="fb19b967-08ca-4541-bffb-aa666f153cda"
# Percentage of people survived in each Pclass
(titanic.groupby("Pclass").mean()["Survived"]*100).plot(kind="bar")
plt.ylabel("% of People Survived")
```

<!-- #region id="31d93044-37eb-4cfc-a725-bb3d21e286a7" -->
## **Pair Plot**
`pairplot` is a function in the Seaborn library for creating a grid of scatterplots that visualize the relationships between pairs of variables in a DataFrame. It's a convenient tool for exploring the pairwise relationships in a dataset, especially when dealing with multiple numerical variables.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="4c2ec4fa-aff0-4538-bfbf-1cd19091ad67" outputId="5bf08490-b25e-4639-ab89-2ba46d623bc6"
iris.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 902} id="110a8cf0-650c-4f74-abe5-303070c03733" outputId="afd859d5-8016-4fde-aeaf-4baa1556a672"
sns.pairplot(iris.iloc[:, 1:], hue="Species")
```

<!-- #region id="PuTSuiYLhtRk" -->
## **Cluster Map**
`clustermap` is a function in the Seaborn library that creates a clustered heatmap, which is a heatmap with dendrograms on both the rows and columns. It is a powerful visualization tool for exploring patterns and relationships in a dataset by hierarchically clustering both the observations (rows) and variables (columns) based on their similarities.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="0tvIjYfUiAIU" outputId="8e6fce4b-de9a-481d-a1a8-7dff14de7023"
# Get the number of people died and survived for each SibSp
pd.crosstab(index=titanic["Parch"], columns=titanic["Survived"])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="SeDa4Kc5iJ92" outputId="b43269ab-25bc-4b2d-b1ab-0100410a8570"
sns.clustermap(pd.crosstab(index=titanic["Parch"], columns=titanic["Survived"]), cmap="Reds",
               annot=True, fmt="")
```

<!-- #region id="KpiKB1pwohQA" -->
🤔 **Note:** `clustermap` is particularly useful for identifying patterns and relationships in datasets with multiple variables, helping to uncover hidden structures in the data. It is a valuable tool in exploratory data analysis and can provide insights into the underlying organization of complex datasets.
<!-- #endregion -->

<!-- #region id="qfw-eQaPkj4_" -->
## **Line Plot (Numerical - Numerical)**

A line plot, also known as a line chart or line graph, is a type of data visualization that displays data points connected by straight line segments. It is commonly used to illustrate the trend or pattern in a dataset over a continuous interval, such as time. Line plots are particularly effective for showing how a variable changes in relation to another variable or over a sequential period.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="TeUARe3Xko93" outputId="258a7319-b6d8-408a-b585-624e2280baeb"
# Print the first 5 rows of the flights data
flights.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 425} id="D0bFhKlxkvJ3" outputId="50332fe2-0a9e-4265-d936-986491e9b4fd"
# Group the number of passengers and calculate total for each year
flights_stat = flights.groupby("year").sum().reset_index()
flights_stat
```

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="KPi-qtLolB-A" outputId="d8ffc22b-1b36-4ece-eb8c-e74e5f758f75"
# Create a simple line plot between years and passengers
sns.lineplot(data=flights_stat, x="year", y="passengers", marker="o")
```

<!-- #region id="s-9aRpDBlHMh" -->
## **Miscellaneous**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 457} id="xIV70ixmlOya" outputId="86ac7d2c-e8a2-44a8-dab6-b2d7b11cc342"
# Create a pivot table
flights.pivot_table(values="passengers", index="month", columns="year")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 488} id="dQ0AU_rglnt2" outputId="3e2500b3-cdc0-49d2-ed39-83f68cefe2c9"
# Create a heatmap
sns.heatmap(flights.pivot_table(values="passengers", index="month", columns="year"),
            cmap="Reds")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="cAhYw43Dm7S1" outputId="a4391b2b-5457-4765-9dfa-09b7b7a7f2f0"
# Create a cluster map
sns.clustermap(flights.pivot_table(values="passengers", index="month", columns="year"),
            cmap="Reds")
```

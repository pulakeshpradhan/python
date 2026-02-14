[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/01_Exploratory_Data_Analysis_(EDA)/01_EDA_using_Bivariate_and_Multivariate_Analysis.ipynb)

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

# **EDA Using Bivariate and Multivariate Analysis**
Exploratory Data Analysis (EDA) is a critical initial step in the data analysis process that involves summarizing, visualizing, and understanding the main characteristics and patterns present in a dataset. Bivariate and multivariate analysis are two important components of EDA that help you explore relationships and interactions between multiple variables in your dataset.


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
# Reading the 'tips' data from seaborn
tips = sns.load_dataset("tips")
```

```python
# Read the 'titanic' data
titanic = pd.read_csv(r"D:\Coding\Datasets\titanic.csv")
```

```python
# Read the 'iris' dataset
iris = pd.read_csv("D:\Coding\Datasets\Iris.csv")
```

```python
# Read the 'flight' data
flights = pd.read_csv(r"D:\Coding\Datasets\flights.csv")
```

```python
tips.head()
```

```python
tips.shape
```

## **Scatter Plot (Numerical - Numerical)**

```python
# Plotting a scatterplot between 'total_bill' and 'tip' using univariate analysis
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.show()
```

```python
# Plotting a scatterplot between 'total_bill' and 'tip' using multivariate analysis
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex", style="smoker", size="size")
plt.legend(bbox_to_anchor=(1, 1), loc=2)
```

## **Bar Plot (Numerical -Categorical)**

```python
titanic.head()
```

```python
# Plot the avergae age of a passsenger traveling in a particular class
sns.barplot(data=titanic, x="Pclass", y="Age")
```

```python
# Plot the avergae age of a passsenger (male and female) traveling in a particular class
sns.barplot(data=titanic, x="Pclass", y="Age", hue="Sex")
```

```python
# Plot the avergae fare in a particular class
sns.barplot(data=titanic, x="Pclass", y="Fare", hue="Sex")
```

## **Box Plot (Numerical - Categorical)**

```python
# Plot a boxplot between 'Sex' and 'Age'
sns.boxplot(data=titanic, x="Sex", y="Age")
```

```python
# Plot a boxplot between 'Sex' and 'Age' also show 'Survived'
sns.boxplot(data=titanic, x="Sex", y="Age", hue="Survived")
```

## **Dist Plot (Numerical - Categorical)**

```python
# Plot the relationship between 'Age' and 'Survived'
# Plotting the Age of the people who could not survive
sns.distplot(titanic[titanic["Survived"]==0]["Age"], hist=False)
# Plotting the Age of the people who could survive
sns.distplot(titanic[titanic["Survived"]==1]["Age"], hist=False)
```

## **Heat Map (Categorical - Categorical)**

```python
titanic.head()
```

```python
# Get the number of people died and survived in each Pclass
pd.crosstab(index=titanic["Pclass"], columns=titanic["Survived"])
```

```python
sns.heatmap(pd.crosstab(index=titanic["Pclass"], columns=titanic["Survived"]), cmap="Reds", 
            annot=True, fmt="")
```

```python
titanic.groupby("Pclass").mean()
```

```python
# Percentage of people survived in each Pclass
(titanic.groupby("Pclass").mean()["Survived"]*100).plot(kind="bar")
plt.ylabel("% of People Survived")
```

## **Cluster Map (Categorical - Categorical)**

```python
# Get the number of people died and survived for each SibSp
pd.crosstab(index=titanic["Parch"], columns=titanic["Survived"])
```

```python
sns.clustermap(pd.crosstab(index=titanic["Parch"], columns=titanic["Survived"]), cmap="Reds", 
               annot=True, fmt="")
```

## **Pair Plot**

```python
iris.head()
```

```python
sns.pairplot(iris.iloc[:, 1:], hue="Species")
```

## **Line Plot (Numerical - Numerical)**

```python
flights.head()
```

```python
# Group the number of passengers and calculate total for each year
flights_stat = flights.groupby("year").sum().reset_index()
flights_stat
```

```python
# Create a simple line plot between years and passengers
sns.lineplot(data=flights_stat, x="year", y="passengers", marker="o")
```

```python
# Create a heatmap 
flights.pivot_table(values="passengers", index="month", columns="year")
```

```python
sns.heatmap(flights.pivot_table(values="passengers", index="month", columns="year"),
            cmap="Reds")
```

```python
sns.clustermap(flights.pivot_table(values="passengers", index="month", columns="year"),
            cmap="Reds")
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/06_Linear_Regression/Simple_Logistic_Regression.ipynb)

# **Simple Logistic Regression (Binary Classification)**
#### **Description:**
Logistic regression is a statistical technique used to analyze the relationship between a dependent variable and one or more independent variables. In binary classification problems, the dependent variable is dichotomous, meaning it has only two possible outcomes. For example, whether a customer will purchase a product or not, whether a person has a certain disease or not, or whether an email is spam or not.

The goal of logistic regression is to find the best model that predicts the probability of an event occurring (in binary classification, the probability of the positive outcome). The logistic regression model uses a function called the sigmoid function, which maps any input value to a value between 0 and 1. This allows the model to output a probability score that can be interpreted as the likelihood of the positive outcome.

### **01. Importing Required Libraries**


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

### **02. Reading the CSV File with Pandas**


```python
# Defining the path of the csv
csv_path = "D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\HR_comma_sep.csv"
```


```python
# Reading the csv file with pandas library
df = pd.read_csv(csv_path)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>Department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the shape of the dataframe
df.shape
```




    (14999, 10)



### **03. Data Cleaning**

#### **3.01 Checking the DataFrame for Null Values**


```python
# Checking the dataframe if there is any null values
df.isnull().sum()
```




    satisfaction_level       0
    last_evaluation          0
    number_project           0
    average_montly_hours     0
    time_spend_company       0
    Work_accident            0
    left                     0
    promotion_last_5years    0
    Department               0
    salary                   0
    dtype: int64



#### **3.02 Checking the datatypes for all the Columns**


```python
df.dtypes
```




    satisfaction_level       float64
    last_evaluation          float64
    number_project             int64
    average_montly_hours       int64
    time_spend_company         int64
    Work_accident              int64
    left                       int64
    promotion_last_5years      int64
    Department                object
    salary                    object
    dtype: object



### **04. Data Exploration and Visualization**

#### **4.01 Plotting the Correlation Matrix of the DataFrame**
Plotting the Correlation Matrix to check which variables have direct and clear impact on employee retention (i.e. whether they leave the company or continue to work) 


```python
# Dropping the columns with qualitative data
data = df.loc[:, "satisfaction_level":"promotion_last_5years"]
# Checking the shape of the data
data.shape
```




    (14999, 8)




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting the correlation matrix
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()
```


    
![png](Simple_Logistic_Regression_files/Simple_Logistic_Regression_16_0.png)
    


**Interpretaion of Correlation Matrix:** <br>
A correlation matrix is a table that displays the pairwise correlations between all the variables in a dataset. Correlation coefficients range from -1 to 1, where a value of -1 indicates a perfect negative correlation, a value of 1 indicates a perfect positive correlation, and a value of 0 indicates no correlation.

In the given dataset, the correlation matrix shows that the 'satisfaction_level' column has the maximum negative correlation coefficient of -0.39 with the "left' column. This means that the satisfaction level of employees has a larger impact on their retention.

#### **4.02 Plotting Bar Charts between Employee Salaries and Retention**
The pandas crosstab() function is used to compute a cross-tabulation table of two or more factors. It is a useful tool in data analysis for examining the relationship between two or more variables, especially when one or both of the variables are categorical.

Here, we are calculating the total number of employees in each salary bracket who stayed or left by summing across the rows using pandas crosstab() function.


```python
# Calculating the total number of employees who stayed or left in each salary bracket
salary_left_data = pd.crosstab(df.salary, df.left)
salary_left_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>left</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>salary</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>high</th>
      <td>1155</td>
      <td>82</td>
    </tr>
    <tr>
      <th>low</th>
      <td>5144</td>
      <td>2172</td>
    </tr>
    <tr>
      <th>medium</th>
      <td>5129</td>
      <td>1317</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Renaming the column names
# 0 = Retained
# 1 = Left
columnNames = {
    0: "Retained",
    1: "Left"
}
salary_left_data.rename(columns=columnNames, inplace=True)
salary_left_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>left</th>
      <th>Retained</th>
      <th>Left</th>
    </tr>
    <tr>
      <th>salary</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>high</th>
      <td>1155</td>
      <td>82</td>
    </tr>
    <tr>
      <th>low</th>
      <td>5144</td>
      <td>2172</td>
    </tr>
    <tr>
      <th>medium</th>
      <td>5129</td>
      <td>1317</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting the bar chart
salary_left_data.plot(kind="bar", color=["green", "red"])
plt.title("Bar Chart Showing Impact of Salaries on Retention")
plt.ylabel("Count")
plt.xlabel("Salary")
```




    Text(0.5, 0, 'Salary')




    
![png](Simple_Logistic_Regression_files/Simple_Logistic_Regression_21_1.png)
    


#### **4.03 Plotting Bar Charts between Departments and Retention**
Here, we are calculating the total number of employees in each department who stayed or left by summing across the rows using pandas crosstab() function.


```python
# Calculating the total number of employees who stayed or left in each department
dept_left_data = pd.crosstab(df.Department, df.left)
dept_left_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>left</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Department</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IT</th>
      <td>954</td>
      <td>273</td>
    </tr>
    <tr>
      <th>RandD</th>
      <td>666</td>
      <td>121</td>
    </tr>
    <tr>
      <th>accounting</th>
      <td>563</td>
      <td>204</td>
    </tr>
    <tr>
      <th>hr</th>
      <td>524</td>
      <td>215</td>
    </tr>
    <tr>
      <th>management</th>
      <td>539</td>
      <td>91</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Renaming the column names
# 0 = Retained
# 1 = Left
dept_left_data.rename(columns=columnNames, inplace=True)
dept_left_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>left</th>
      <th>Retained</th>
      <th>Left</th>
    </tr>
    <tr>
      <th>Department</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IT</th>
      <td>954</td>
      <td>273</td>
    </tr>
    <tr>
      <th>RandD</th>
      <td>666</td>
      <td>121</td>
    </tr>
    <tr>
      <th>accounting</th>
      <td>563</td>
      <td>204</td>
    </tr>
    <tr>
      <th>hr</th>
      <td>524</td>
      <td>215</td>
    </tr>
    <tr>
      <th>management</th>
      <td>539</td>
      <td>91</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting the bar chart
dept_left_data.plot(kind="bar", color=["green", "red"])
plt.title("Bar Chart Showing Correlaton between Departments and Employee Retention")
plt.xlabel("Department")
plt.ylabel("Count")
plt.show()
```


    
![png](Simple_Logistic_Regression_files/Simple_Logistic_Regression_25_0.png)
    


#### **4.04 Plotting the Scatterplot between 'satisfaction_level' and 'left'**


```python
# Extracting a small sample from the dataframe to represent the scatterplot
test_df = df.sample(50, random_state=75)
```


```python
# Plotting the Scatterplot between 'satisfaction_level' and 'left'
sns.scatterplot(x=test_df["satisfaction_level"], y=test_df["left"], color="red")
plt.title("Scatterplot between 'satisfaction_level' and 'left'")
plt.ylabel("Employee Retention (0=Retained, 1=Left)")
plt.xlabel("Satisfaction Level")
plt.grid()
plt.show()
```


    
![png](Simple_Logistic_Regression_files/Simple_Logistic_Regression_28_0.png)
    


### **05. Dividing the Data into Training and Testing Set**

#### **5.01 Defifining the Dependent and Independent Variable**


```python
# Dependent Variable (y) = "left"
# Independent Variable (x) = "satisfaction_level"
x = df[["satisfaction_level"]]
y = df[["left"]]
```

#### **5.02 Splitting the Data into Training and Testing Set**


```python
# Importing the train_test_split from sklearn library
from sklearn.model_selection import train_test_split
```


```python
# Training data = 70% and Testing data = 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=75)
```

### **06. Instantiating the Simple Logistic Regression Model**

#### **6.01 Importing Logistic Regression Model from sklearn Library**


```python
# Importing the Logistic Regression Model from sklearn library
from sklearn.linear_model import LogisticRegression
```

#### **6.02 Generating a Logistic Regression Object**


```python
# Creating a linear regression object
log_reg = LogisticRegression()
# Feeding the training data to the model
log_reg.fit(x_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>



#### **6.03 Getting the Coefficients of the Linear Regression Model**


```python
# Getting the slope of the model
log_reg.coef_
```




    array([[-3.84563398]])




```python
# Getting the y-intercept of the model
log_reg.intercept_
```




    array([0.97025215])



### **07. Validation of the Model**

#### **7.01 Validating the Logistic Regression Model**


```python
# Predicting the left status of the x_test (satisfaction_level) data
y_predict = log_reg.predict(x_test)
y_predict
```




    array([0, 0, 1, ..., 0, 0, 0], dtype=int64)




```python
# Getting the accuracy score of the model
log_reg.score(x_test, y_test)
```




    0.7624444444444445




```python
# Getting the prediction probability of the x_test data
log_reg.predict_proba(x_test)
```




    array([[0.9118948 , 0.0881052 ],
           [0.62036288, 0.37963712],
           [0.35762467, 0.64237533],
           ...,
           [0.61126493, 0.38873507],
           [0.67300916, 0.32699084],
           [0.93368009, 0.06631991]])



# **Simple Linear Regression**
#### **Description:** 
Simple linear regression is a statistical technique used to establish a relationship between two variables - one independent and one dependent. The purpose of this technique is to determine whether there is a linear relationship between the two variables, and if so, to develop a mathematical model that can be used to predict the value of the dependent variable based on the value of the independent variable.
#### **Assumptions:** 
Simple linear regression is a parametric test, meaning that it makes certain assumptions about the data. These assumptions are:
1. **Homogeneity of variance:** the size of the error in our prediction doesn’t change significantly across the values of the independent variable.
2. **Independence of observations:** the observations in the dataset were collected using statistically valid sampling methods, and there are no hidden relationships among observations.
3. **Normality:** The data follows a normal distribution.

#### **How to perform a Simple Linear Regression:**
**Simple linear regression Formula:**
$$ y = {\beta_0} + {\beta_1{X}} + {\epsilon} $$
* y is the predicted value of the dependent variable (y) for any given value of the independent variable (x).
* B0 is the intercept, the predicted value of y when the x is 0.
* B1 is the regression coefficient or slope – how much we expect y to change as x increases.
* x is the independent variable ( the variable we expect is influencing y).
* e is the error of the estimate, or how much variation there is in our estimate of the regression coefficient.

Linear regression finds the line of best fit line through your data by searching for the regression coefficient (B1) that minimizes the total error (e) of the model.

## **Simple Linear Regression Project:** 
### **Predicting Average Temperature in Northern Hemisphere based on Latitude using Simple Linear Regression**
#### **Project Description:**
The relationship between latitude and temperature is a well-established phenomenon in climatology. In general, temperature decreases as we move away from the equator towards the poles. This is because the Earth's surface receives more direct sunlight near the equator than at the poles, and therefore the equator is warmer.

The project of predicting average temperature based on latitude using simple linear regression is an exciting machine learning endeavor that has many real-world applications. The goal of this project is to build a model that can accurately estimate the average temperature of a particular location based on its latitude information.

### **01. Importing Required Libraries**


```python
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
```

### **02. Reading the CSV File with Pandas**


```python
# Defining the path of the csv
csv_path = "D:\Coding\Git Repository\Data-Science-Bootcamp-with-Python\Datasets\GlobalLandTemperaturesByCity.csv"
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
      <th>Unnamed: 0</th>
      <th>dt</th>
      <th>AverageTemperature</th>
      <th>AverageTemperatureUncertainty</th>
      <th>City</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1881-01-01</td>
      <td>7.858</td>
      <td>0.962</td>
      <td>Nuevo Laredo</td>
      <td>United States</td>
      <td>28.13N</td>
      <td>99.09W</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1962-09-01</td>
      <td>21.115</td>
      <td>0.309</td>
      <td>Musoma</td>
      <td>Tanzania</td>
      <td>0.80S</td>
      <td>34.55E</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1841-09-01</td>
      <td>11.590</td>
      <td>1.466</td>
      <td>Lyubertsy</td>
      <td>Russia</td>
      <td>55.45N</td>
      <td>36.85E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1972-06-01</td>
      <td>24.751</td>
      <td>0.386</td>
      <td>João Pessoa</td>
      <td>Brazil</td>
      <td>7.23S</td>
      <td>34.86W</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1915-04-01</td>
      <td>26.726</td>
      <td>0.935</td>
      <td>Carmen</td>
      <td>Mexico</td>
      <td>18.48N</td>
      <td>91.27W</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the shape of the dataframe
df.shape
```




    (20000, 8)



### **03. Data Cleaning**

#### **3.01 Removing the Rows with Missing Values**


```python
# Counting the number of missing values (i.e., NaN values) in each column of the pandas DataFrame
df.isnull().sum()
```




    Unnamed: 0                       0
    dt                               0
    AverageTemperature               0
    AverageTemperatureUncertainty    0
    City                             0
    Country                          0
    Latitude                         0
    Longitude                        0
    dtype: int64




```python
# Dropping the rows with missing values
df.dropna(inplace=True)
# Checking the shape of the dataframe after dropping rows with missing values
df.shape
```




    (20000, 8)




```python
# Checking the dataframe after dropping the rows with missing values
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
      <th>Unnamed: 0</th>
      <th>dt</th>
      <th>AverageTemperature</th>
      <th>AverageTemperatureUncertainty</th>
      <th>City</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1881-01-01</td>
      <td>7.858</td>
      <td>0.962</td>
      <td>Nuevo Laredo</td>
      <td>United States</td>
      <td>28.13N</td>
      <td>99.09W</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1962-09-01</td>
      <td>21.115</td>
      <td>0.309</td>
      <td>Musoma</td>
      <td>Tanzania</td>
      <td>0.80S</td>
      <td>34.55E</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1841-09-01</td>
      <td>11.590</td>
      <td>1.466</td>
      <td>Lyubertsy</td>
      <td>Russia</td>
      <td>55.45N</td>
      <td>36.85E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1972-06-01</td>
      <td>24.751</td>
      <td>0.386</td>
      <td>João Pessoa</td>
      <td>Brazil</td>
      <td>7.23S</td>
      <td>34.86W</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1915-04-01</td>
      <td>26.726</td>
      <td>0.935</td>
      <td>Carmen</td>
      <td>Mexico</td>
      <td>18.48N</td>
      <td>91.27W</td>
    </tr>
  </tbody>
</table>
</div>



#### **3.02 Manipulating the 'Latitude' Column**


```python
# Adding a "-" (minus) before the latitudes of the sothern hemisphere
df["Latitude"] = df["Latitude"].apply(lambda x: "-"+x if x.endswith("S") else x)
```


```python
# Removing the 'N' and 'S' from 'Latitude' column
df["Latitude"] = df["Latitude"].str.replace("N", "")
df["Latitude"] = df["Latitude"].str.replace("S", "")
```


```python
# Checking the dataframe
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
      <th>Unnamed: 0</th>
      <th>dt</th>
      <th>AverageTemperature</th>
      <th>AverageTemperatureUncertainty</th>
      <th>City</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1881-01-01</td>
      <td>7.858</td>
      <td>0.962</td>
      <td>Nuevo Laredo</td>
      <td>United States</td>
      <td>28.13</td>
      <td>99.09W</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1962-09-01</td>
      <td>21.115</td>
      <td>0.309</td>
      <td>Musoma</td>
      <td>Tanzania</td>
      <td>-0.80</td>
      <td>34.55E</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1841-09-01</td>
      <td>11.590</td>
      <td>1.466</td>
      <td>Lyubertsy</td>
      <td>Russia</td>
      <td>55.45</td>
      <td>36.85E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1972-06-01</td>
      <td>24.751</td>
      <td>0.386</td>
      <td>João Pessoa</td>
      <td>Brazil</td>
      <td>-7.23</td>
      <td>34.86W</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1915-04-01</td>
      <td>26.726</td>
      <td>0.935</td>
      <td>Carmen</td>
      <td>Mexico</td>
      <td>18.48</td>
      <td>91.27W</td>
    </tr>
  </tbody>
</table>
</div>



####  **3.03 Changing the Datatype of the 'Latitude' Column**


```python
# Changing the datatype of the 'Latitude' column from 'str' to 'float'
convert_dict = {"Latitude": float}
df = df.astype(convert_dict)
df.dtypes
```




    Unnamed: 0                         int64
    dt                                object
    AverageTemperature               float64
    AverageTemperatureUncertainty    float64
    City                              object
    Country                           object
    Latitude                         float64
    Longitude                         object
    dtype: object




```python
# Checking the latitudes of the southern hemisphere
df[df["Latitude"] < 0].head()
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
      <th>Unnamed: 0</th>
      <th>dt</th>
      <th>AverageTemperature</th>
      <th>AverageTemperatureUncertainty</th>
      <th>City</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1962-09-01</td>
      <td>21.115</td>
      <td>0.309</td>
      <td>Musoma</td>
      <td>Tanzania</td>
      <td>-0.80</td>
      <td>34.55E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1972-06-01</td>
      <td>24.751</td>
      <td>0.386</td>
      <td>João Pessoa</td>
      <td>Brazil</td>
      <td>-7.23</td>
      <td>34.86W</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>1965-07-01</td>
      <td>20.737</td>
      <td>0.402</td>
      <td>Bukavu</td>
      <td>Congo (Democratic Republic Of The)</td>
      <td>-2.41</td>
      <td>28.13E</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>1943-06-01</td>
      <td>22.852</td>
      <td>0.513</td>
      <td>Toliary</td>
      <td>Madagascar</td>
      <td>-23.31</td>
      <td>42.82E</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>1985-06-01</td>
      <td>15.466</td>
      <td>0.427</td>
      <td>Ferraz De Vasconcelos</td>
      <td>Brazil</td>
      <td>-23.31</td>
      <td>46.31W</td>
    </tr>
  </tbody>
</table>
</div>



#### **3.04 Selecting the Northern Latitudes Only**


```python
# Selecting the latitudes of the Northern Hemisphere only
df = df[df["Latitude"] >= 0]
```

#### **3.05 Selecting a Random Sample from the DataFrame**


```python
# Selecting a random sample from the dataframe
df = df.sample(10000, random_state=0)
```


```python
# Selecting only two columns 'Latitude' and 'AverageTemperature' from the dataframe
df = df[["Latitude", "AverageTemperature"]]
```

#### **3.06 Removing the Outliers**


```python
# Visualizing the boxplot of the dataframe
sns.boxplot(df)
plt.title("Boxplot before Outliers Removal")
```




    Text(0.5, 1.0, 'Boxplot before Outliers Removal')




    
![png](01_Simple_Linear_Regression_files/01_Simple_Linear_Regression_26_1.png)
    



```python
# Removing the outliers from the 'AverageTemperature' column
# Getting the value of First and Third Quartile (Q1 & Q3) of the 'AverageTemperature'
Q1 = df["AverageTemperature"].quantile(0.25)
Q3 = df["AverageTemperature"].quantile(0.75)
print("Quartile1:", Q1)
print("Quartile2:", Q3)
```

    Quartile1: 8.628
    Quartile2: 25.2815
    


```python
# Calulating the Inter Quartile Range (IQR)
IQR = Q3 - Q1
print("IQR:", IQR)
```

    IQR: 16.6535
    


```python
# Calculating the Higher Fence and Lower Fence
lower_fence =  Q1 - (1.5 * IQR)
higher_fence = Q3 + (1.5 * IQR)
print("Lower Fence:", lower_fence)
print("Higher Fence:", higher_fence)
```

    Lower Fence: -16.35225
    Higher Fence: 50.261750000000006
    


```python
# Removing the Outliers
df = df[~((df["AverageTemperature"] < lower_fence) | (df["AverageTemperature"] > higher_fence))]
```

#### **3.07 Checking the Final DataFrame**


```python
df.shape
```




    (9927, 2)




```python
# Resetting the index of the dataframe
df.reset_index(drop=True, inplace=True)
```


```python
# Checking the dataframe
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
      <th>Latitude</th>
      <th>AverageTemperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.95</td>
      <td>20.134</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36.17</td>
      <td>12.346</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47.42</td>
      <td>-0.779</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36.17</td>
      <td>15.944</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40.99</td>
      <td>0.793</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Describing the univariate statistics of the dataframe
df.describe()
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
      <th>Latitude</th>
      <th>AverageTemperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9927.000000</td>
      <td>9927.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>33.014089</td>
      <td>16.345360</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.580933</td>
      <td>10.448532</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.800000</td>
      <td>-16.335000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.310000</td>
      <td>8.856500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>34.560000</td>
      <td>18.094000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>44.200000</td>
      <td>25.317500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>69.920000</td>
      <td>37.283000</td>
    </tr>
  </tbody>
</table>
</div>



### **04. Data Visualization**

#### **4.01 Histogram of Average Temperature with Kernel Density Estimation (KDE))**


```python
# Visualizing Histogram of 'AverageTemperature' with Probability Density Function
sns.histplot(df["AverageTemperature"], kde=True)
plt.title("Historam of Average Temperature")
plt.show()
```


    
![png](01_Simple_Linear_Regression_files/01_Simple_Linear_Regression_38_0.png)
    


#### **4.02 Boxplot of Average Temperature**


```python
# Visualizing 5 number summary of the 'AverageTemperature'
sns.boxplot(x=df["AverageTemperature"], width=0.5)
plt.title("Boxplot of Average Temperature after Outliers Removal")
plt.show()
```


    
![png](01_Simple_Linear_Regression_files/01_Simple_Linear_Regression_40_0.png)
    


#### **4.03 Scatterplot between Latitude and Average Temperature**


```python
sns.scatterplot(x=df["Latitude"], y=df["AverageTemperature"], marker=".")
plt.title("Scatterplot between North Latitude and Average Temperature")
plt.xlabel("North Latitude")
plt.ylabel("Average Temperature (in °C)")
plt.show()
```


    
![png](01_Simple_Linear_Regression_files/01_Simple_Linear_Regression_42_0.png)
    


### **05. Dividing the Data into Trainining and Testing Set**

#### **5.01 Defining the Dependent and Independent Variable**


```python
# Dependent Variable (y) = 'AverageTemperature'
# Independent Variable (x) = 'Latitude'
x = df[["Latitude"]]
y = df[["AverageTemperature"]]
```

#### **5.02 Splitting the Data into Training and Testing Set**


```python
# Importing the tarin_test_split from sklearn library
from sklearn.model_selection import train_test_split
```


```python
# Training Data = 70% and Testing Data = 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=75)
```

### **06. Instantiating the Simple Linear Regression Model**

#### **6.01 Importing LinearRegression Model from sklearn Library**


```python
# Importing the Linear Regression Model from sklearn library
from sklearn.linear_model import LinearRegression
```

#### **6.02 Generating a LinearRegression Object**


```python
# Creating a linear regression object
lin_reg = LinearRegression()
# Feeding the training data to the model
lin_reg.fit(x_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>



#### **6.03 Getting the Coefficients of the Linear Regression Model**


```python
# Getting the slope of the model
lin_reg.coef_
```




    array([[-0.48602283]])




```python
# Getting the y-intercept of the model
lin_reg.intercept_
```




    array([32.37125607])



### **07. Validation of the Model**

#### **7.01 Importing Some Validation Metrics**


```python
# Importing some validation metrics from sklearn library
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
```

#### **7.02 Validating the Linear Regression Model**


```python
# Predicting the AverageTemperature of the x_test (Latitude) data
y_predicted = lin_reg.predict(x_test)
```


```python
# Defining the actual 'Average Temperature' data
y_actual = y_test
```

#### **7.03 Calculating the Mean Absolute Error, Mean Squared Error of the Model**


```python
# Calculating the Mean Absolute Error (MAE)
MAE = mean_absolute_error(y_actual, y_predicted)
print("Mean Absolute Error (MAE):", MAE.round(4))
```

    Mean Absolute Error (MAE): 6.0816
    


```python
# Calulating the Mean Squared Error (MSE)
MSE = mean_squared_error(y_actual, y_predicted)
print("Mean Squared Error (MSE):", MSE.round(4))
```

    Mean Squared Error (MSE): 57.4893
    


```python
# Calulating the Root Mean Squared Error (MSE)
RMSE = np.sqrt(MSE)
print("Root Mean Squared Error (RMSE):", RMSE.round(4))
```

    Root Mean Squared Error (RMSE): 7.5822
    

#### **7.04 Plotting the Linear Regression Line**


```python
sns.scatterplot(x=df["Latitude"], y=df["AverageTemperature"], marker=".")
plt.plot(x, lin_reg.predict(x), color="red", label="Regression Line")
plt.title("Scatterplot between North Latitude and Average Temperature")
plt.xlabel("North Latitude")
plt.ylabel("Average Temperature (in °C)")
plt.legend()
plt.show()
```


    
![png](01_Simple_Linear_Regression_files/01_Simple_Linear_Regression_68_0.png)
    


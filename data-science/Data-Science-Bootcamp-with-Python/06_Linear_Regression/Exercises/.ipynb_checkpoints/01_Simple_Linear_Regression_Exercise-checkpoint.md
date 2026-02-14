## 01. Importing Required Libraries


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
%matplotlib inline
```

## 02. Reading the CSV with Pandas


```python
df = pd.read_csv("GDP_per_capita_usa.csv")
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
      <th>Year</th>
      <th>GDP</th>
      <th>Per Capita</th>
      <th>Growth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>$23,315.08B</td>
      <td>$70,249</td>
      <td>5.95%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>$21,060.47B</td>
      <td>$63,531</td>
      <td>-2.77%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>$21,380.98B</td>
      <td>$65,120</td>
      <td>2.29%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018</td>
      <td>$20,533.06B</td>
      <td>$62,823</td>
      <td>2.95%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>$19,477.34B</td>
      <td>$59,908</td>
      <td>2.24%</td>
    </tr>
  </tbody>
</table>
</div>



## 03. Data Preprocessing


```python
# Removing the dollar, comma and percentage symbol from the dataframe
df["GDP"] = df["GDP"].str.replace("$", "")
df["GDP"] = df["GDP"].str.replace(",", "")
df["GDP"] = df["GDP"].str.replace("B", "")
df["Per Capita"] = df["Per Capita"].str.replace("$", "")
df["Per Capita"] = df["Per Capita"].str.replace(",", "")
df["Growth"] = df["Growth"].str.replace("%", "")
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
      <th>Year</th>
      <th>GDP</th>
      <th>Per Capita</th>
      <th>Growth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>23315.08</td>
      <td>70249</td>
      <td>5.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>21060.47</td>
      <td>63531</td>
      <td>-2.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>21380.98</td>
      <td>65120</td>
      <td>2.29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018</td>
      <td>20533.06</td>
      <td>62823</td>
      <td>2.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>19477.34</td>
      <td>59908</td>
      <td>2.24</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the datatypes of the columns
df.dtypes
```




    Year           int64
    GDP           object
    Per Capita    object
    Growth        object
    dtype: object




```python
# Using a dictionary to change the datatype of columns from object to int and float
dt_convert = {
    "GDP": float,
    "Per Capita": float,
}
```


```python
# Changing the datatypes of the columns
df = df.astype(dt_convert)
df.dtypes
```




    Year            int64
    GDP           float64
    Per Capita    float64
    Growth         object
    dtype: object




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
      <th>Year</th>
      <th>GDP</th>
      <th>Per Capita</th>
      <th>Growth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>23315.08</td>
      <td>70249.0</td>
      <td>5.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>21060.47</td>
      <td>63531.0</td>
      <td>-2.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>21380.98</td>
      <td>65120.0</td>
      <td>2.29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018</td>
      <td>20533.06</td>
      <td>62823.0</td>
      <td>2.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>19477.34</td>
      <td>59908.0</td>
      <td>2.24</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Evaluating the general statistics of the dataframe
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
      <th>Year</th>
      <th>GDP</th>
      <th>Per Capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1990.500000</td>
      <td>7935.587903</td>
      <td>27417.145161</td>
    </tr>
    <tr>
      <th>std</th>
      <td>18.041619</td>
      <td>6738.805659</td>
      <td>20172.070442</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1960.000000</td>
      <td>543.300000</td>
      <td>3007.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1975.250000</td>
      <td>1732.027500</td>
      <td>7998.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1990.500000</td>
      <td>6060.635000</td>
      <td>24115.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2005.750000</td>
      <td>13621.492500</td>
      <td>45757.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2021.000000</td>
      <td>23315.080000</td>
      <td>70249.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 04. Create a Scatterplot between Year and GDP Per Capita


```python
# Selecting independent(x) and dependent variable(y)
# independent variable = Year
# dependent variable = GDP Per Capita (US $)
x = df[["Year"]].values
y = df[["Per Capita"]].values
```


```python
plt.plot(x, y)
plt.title("USA GDP Per Capita (US$)")
plt.xlabel("Year")
plt.ylabel("GDP Per Capita (US$)")
plt.show()
```


    
![png](01_Simple_Linear_Regression_Exercise-checkpoint_files/01_Simple_Linear_Regression_Exercise-checkpoint_14_0.png)
    


## 05. Creating a Linear Regression Object


```python
lin_reg = linear_model.LinearRegression()
```


```python
# Training the Linear Regression model
lin_reg.fit(x, y)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>



## 06. Accessing the Slope and Intercept
Linear Equation: <br>
**y = mx + c**<br>
where,<br>
	y = independent variable<br>
    m = slope<br>
    x = dependent variable<br>
    c = intercept<br>


```python
slope = lin_reg.coef_
print("Slope(m):", slope)
```

    Slope(m): [[1099.76930825]]
    


```python
intercept = lin_reg.intercept_
print("intercept(c):", intercept)
```

    intercept(c): [-2161673.66291456]
    

## 07. Predicting the GDP of Year 2030, 2040 and 2050


```python
lin_reg.predict([[2030], [2040], [2050]])
```




    array([[70858.03283725],
           [81855.72591977],
           [92853.41900229]])



## Assessing the Accuracy of the Linear Regression Model


```python
# Predicting the GDP of all the years in the dataframe
predicted_gdp = lin_reg.predict(x)
```


```python
# Plotting the Regression Line
plt.plot(x, y)
plt.plot(x, predicted_gdp, linestyle="--", linewidth=1)
plt.title("USA GDP Per Capita (US$)")
plt.xlabel("Year")
plt.ylabel("GDP Per Capita (US$)")
plt.show()
```


    
![png](01_Simple_Linear_Regression_Exercise-checkpoint_files/01_Simple_Linear_Regression_Exercise-checkpoint_25_0.png)
    


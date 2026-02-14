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
<a href="https://colab.research.google.com/github/geonextgis/Mastering-Machine-Learning-and-GEE-for-Earth-Science/blob/main/04_Machine_Learning_Algorithms/02_Regression_Metrics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="9aaa40a1-1804-4cde-9605-4cdadc610993" -->
# **Regression Metrics**
Regression metrics are used to evaluate the performance of predictive models that aim to estimate a continuous target variable. These metrics help assess how well the model's predictions align with the actual values, allowing you to understand the accuracy, precision, and goodness of fit of your regression model. Here are some commonly used regression metrics:
<!-- #endregion -->

<!-- #region id="75a46037-64d2-4a00-8644-049f09e20bf4" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7XDqnfZD-AKe" outputId="d178172b-c2fd-4b70-99fd-4a09b1492261"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="92fa9d73-a883-461a-be23-6ff065abd927"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")
```

<!-- #region id="cf258036-b9a6-4949-97e6-2b442098f30c" -->
## **Read the Data**
<!-- #endregion -->

```python id="0c90745f-be6c-4520-a1ef-1d1aaab8e30a" outputId="a9b40ed5-1244-4c2f-d003-32796b27b8c3" colab={"base_uri": "https://localhost:8080/", "height": 206}
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GitHub Repo/Mastering-Machine-Learning-and-GEE-for-Earth-Science/Datasets/Placement_SLR.csv")
df.head()
```

```python id="71b20f08-238d-4103-8a81-0018d371d418" outputId="adf8d71d-ea83-48e3-9142-af4b3117623f" colab={"base_uri": "https://localhost:8080/"}
# Check for the null values
df.isnull().sum()
```

<!-- #region id="6b51e43c-96ae-4dc0-ba9c-8c098a01c92a" -->
## **Train Test Split**
<!-- #endregion -->

```python id="0d8d40ae-1e86-431b-bdf8-e1a047659f11" outputId="a5781459-fb88-4f27-f00c-5217595aa6b3" colab={"base_uri": "https://localhost:8080/"}
X_train, X_test, y_train, y_test = train_test_split(df.drop("package", axis=1),
                                                    df["package"],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape , X_test.shape
```

<!-- #region id="2efd8c32-fe7d-4674-8853-fc0008b6b88b" -->
## **Plot the Data**
<!-- #endregion -->

```python id="26fd9ae8-e051-4c95-a84e-fb65e532ce8c" outputId="3c760943-94cb-495b-bc89-fc062d3c9dcd" colab={"base_uri": "https://localhost:8080/", "height": 472}
sns.scatterplot(x=X_train["cgpa"], y=y_train)
plt.title("Scatterplot between CGPA and Package")
plt.show()
```

<!-- #region id="a1cadfec-6964-442e-bac9-b098789926f5" -->
## **Train a Simple Linear Regression Model**
<!-- #endregion -->

```python id="4df83756-2eb4-489c-b513-755c8228e9e8" outputId="01fc8a15-5896-45f2-eefb-96212fe4f791" colab={"base_uri": "https://localhost:8080/", "height": 74}
# Instantiate an object of the LinearRegression class
lr = LinearRegression()

# Fit the training data
lr.fit(X_train, y_train)
```

```python id="ac6c7219-eddf-474c-9764-c6a2c312a95d" outputId="012abb3d-44cd-4076-da64-7e6366fd09a0" colab={"base_uri": "https://localhost:8080/"}
# Predict the test data
y_pred = lr.predict(X_test)
y_pred
```

<!-- #region id="8d046beb-61a9-4288-a720-85e4a8368a70" -->
## **Plot the Regression Line**
<!-- #endregion -->

```python id="866160c1-d413-46b0-a1b7-6f62ae7bd853" outputId="c9a526fe-e518-44b8-990e-a0dd87fffcc1" colab={"base_uri": "https://localhost:8080/", "height": 472}
sns.scatterplot(x=X_train["cgpa"], y=y_train)
sns.lineplot(x=X_test["cgpa"], y=lr.predict(X_test), c="red", label="Regression Line")
plt.title("Scatterplot between CGPA and Package")
plt.show()
```

<!-- #region id="bc21deeb-702b-405a-bfca-b8d739d6fb06" -->
## **Check the Accuracy using Regression Metrics**
<!-- #endregion -->

<!-- #region id="4e3945c7-3068-46cd-be62-7bdc17c80740" -->
### **Mean Absolute Error (MAE):**
MAE is the average of the absolute differences between the predicted and actual values. It measures the average magnitude of errors and is easy to understand.

**Advantages:**
* Easy to interpret as it represents the average absolute error.
* Resistant to outliers, as it does not square errors.

**Disadvantages:**
* Does not penalize larger errors more heavily.
* May not work well if the error distribution is not symmetric.

**Formula:**<br>
<center><img src="https://editor.analyticsvidhya.com/uploads/42439Screenshot%202021-10-26%20at%209.34.08%20PM.png" width="40%"></center>
<!-- #endregion -->

```python id="08dfb84d-645b-4c15-88a7-2d30096ce3d3" outputId="81c25808-f665-40e3-e24e-9bbf7defd170" colab={"base_uri": "https://localhost:8080/"}
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae.round(2))
```

<!-- #region id="4b34ebba-d396-4f10-8f50-75492c4b438f" -->
### **Mean Squared Error (MSE):**
MSE measures the average of the squared differences between the predicted and actual values. It gives more weight to larger errors and penalizes them.

**Advantages:**
* Provides a measure of how well the model performs while penalizing larger errors.
* Mathematically convenient and commonly used in optimization algorithms.

**Disadvantages:**
* The squared nature of the metric makes it sensitive to outliers.
* The units of MSE are not the same as the target variable, making it less interpretable.

**Formula:**<br>
<center><img src="https://cdn-media-1.freecodecamp.org/images/hmZydSW9YegiMVPWq2JBpOpai3CejzQpGkNG" width="40%"> </center>
<!-- #endregion -->

```python id="b7e7a16c-3857-40c5-a624-e53725d30e6b" outputId="7b324f5a-a2e3-4166-ff4c-65799a48a92c" colab={"base_uri": "https://localhost:8080/"}
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse.round(2))
```

<!-- #region id="d37a581c-b5b4-4e52-bf46-b6fa31da7e20" -->
### **Root Mean Squared Error (RMSE):**
RMSE is the square root of the MSE. It provides a more interpretable metric in the same units as the target variable.

**Advantages:**
* Shares the same unit as the target variable, which makes it more interpretable than MSE.
* Balances the sensitivity to outliers found in MSE.

**Disadvantages:**
* Like MSE, it can still be sensitive to outliers.
* Not as intuitive as MAE.

**Formula:**<br>
<center><img src="https://miro.medium.com/v2/resize:fit:966/1*lqDsPkfXPGen32Uem1PTNg.png" width="40%"> </center>
<!-- #endregion -->

```python id="000db115-7940-4475-b8dd-274b706a9e23" outputId="31558fb9-5372-4fb4-bad5-473ce0ccc718" colab={"base_uri": "https://localhost:8080/"}
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse.round(2))
```

<!-- #region id="8035fa4b-9d22-443b-8bbe-42da5e1884f9" -->
### **R-squared (R²):**
R-squared measures the proportion of the variance in the target variable that is explained by the model. It ranges from 0 to 1, with higher values indicating a better fit.

**Advantages:**
* Provides a measure of goodness of fit, indicating how well the model explains the variance in the data.
* Values range from 0 to 1, where higher values suggest a better fit.

**Disadvantages:**
* It may increase when adding more predictors, even if they are irrelevant (overfitting).
* R-squared alone doesn't reveal the direction or magnitude of individual errors.

**Formula:**<br>
<center><img src="https://i0.wp.com/www.fairlynerdy.com/wp-content/uploads/2017/01/r_squared_5.png?resize=625%2C193" width="40%"> </center>

<center><img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*EjMnICEPkm0VzoLetqeYbw.jpeg" width="60%"> </center>
<!-- #endregion -->

```python id="a1f28b62-2f25-4fdc-9d0d-a6547bd72e4f" outputId="6268239b-1947-48c2-f131-a77de12fddf3" colab={"base_uri": "https://localhost:8080/"}
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2.round(2))
```

<!-- #region id="5e67c7ac-44b3-4361-92d1-6ecb99381664" -->
### **Adjusted R-squared:**
Adjusted R-squared is a modified version of R-squared that accounts for the number of predictors in the model. It penalizes the addition of irrelevant predictors.

**Advantages:**
* Adjusts R-squared to account for the number of predictors, helping to mitigate overfitting.
* Offers a more realistic assessment of model fit in multiple regression.

**Disadvantages:**
* It can still be influenced by outliers and unrepresentative samples.

**Formula:**<br>
<center><img src="https://i.stack.imgur.com/RcGf6.png" width="50%"> </center>
<!-- #endregion -->

```python id="8e5d2a56-bcc8-4603-948c-f0958f2c56c0"
n = len(X_test) # Total Sample Size
p = len(X_test.columns) # Number of independent variable
```

```python id="5410b9bc-be16-4b66-8faa-812635d462ef" outputId="c7396f77-113c-41c5-95a3-0692c8061be6" colab={"base_uri": "https://localhost:8080/"}
adusted_r2 = 1 - ((1 - r2)*(n - 1) / (n - p - 1))
print("Adjusted R2 Score:", adusted_r2.round(2))
```

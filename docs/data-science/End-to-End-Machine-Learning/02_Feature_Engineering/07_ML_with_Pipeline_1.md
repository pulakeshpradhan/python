[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/07_ML_with_Pipeline_1.ipynb)

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

# **Machine Learning with Pipeline - 1**
Machine Learning with a pipeline is a common practice in the field of data science and machine learning. A pipeline is a series of data processing components (transformers and an estimator) that are chained together to streamline the workflow in machine learning tasks. It helps in organizing and automating the various steps involved in building and evaluating machine learning models.


<img src="https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-pipeline2.png" style="width:100%">


## **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## **Read the Data**

```python
df = pd.read_csv(r"D:\Coding\Datasets\titanic.csv")
df
```

```python
# Dropping the unecessary columns
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
df.head()
```

```python
# Check to column informations
df.info()
```

## **Train Test Split**

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop("Survived", axis=1),
                                                    df["Survived"],
                                                    test_size=0.3,
                                                    random_state=0)
x_train.shape, x_test.shape
```

```python
x_train
```

## **Preprocess the Data using Column Transformer**

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
```

```python
# Create an imputation transformer for 'Age' and 'Embarked' columns
transformer_1 = ColumnTransformer([
    ("impute_age", SimpleImputer(), [2]),
    ("impute_embarked", SimpleImputer(strategy="most_frequent"), [6])
], remainder="passthrough")
```

```python
# Create an One Hot Encoding tranformer for 'Sex' and 'Embarked' columns
transformer_2 = ColumnTransformer([
    ("ohe_sex_embarked", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1, 6])
], remainder="passthrough")
```

```python
# Create a transformer for scale the values
transformer_3 = ColumnTransformer([
    ("scale", MinMaxScaler(), slice(0, 10))
])
```

```python
# Create a transformer to select best 8 features
transformer_4 = SelectKBest(score_func=chi2, k=8)
```

## **Train a Decision Tree Model**

```python
from sklearn.tree import DecisionTreeClassifier
```

```python
# Train a decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=0)
```

## **Create Pipeline**

```python
from sklearn.pipeline import Pipeline
```

```python
pipe = Pipeline([
    ("transformer_1", transformer_1),
    ("transformer_2", transformer_2),
    ("transformer_3", transformer_3),
    ("transformer_4", transformer_4),
    ("transformer_5", dt_classifier)
])
```

## **Pipeline vs make_pipeline**

```python
from sklearn.pipeline import make_pipeline
```

```python
# Alternate Syntax
pipe = make_pipeline(transformer_1, transformer_2, transformer_3, transformer_4, dt_classifier)
```

## **Train the Model using Pipeline**

```python
# Train the model
pipe.fit(x_train, y_train)
```

## **Explore the Pipeline**

```python
# Print the steps in the pipeline
pipe.named_steps
```

```python
# Check the mean value of the SimpleImputer object for 'age' column
pipe.named_steps["columntransformer-1"].transformers_[0][1].statistics_
```

## **Accuracy Assessment**

```python
# Predict the test data
y_pred = pipe.predict(x_test)
```

```python
from sklearn.metrics import accuracy_score
```

```python
# Print the overall accuracy of the model
accuracy_score(y_test, y_pred)
```

## **Cross Validation using Pipeline**

```python
from sklearn.model_selection import cross_val_score
```

```python
# Cross validation using cross_val_score
cross_val_score(pipe, x_train, y_train, cv=5, scoring="accuracy").mean()
```

## **GridSearch using Pipeline**

```python
# Define the parameters for GridSearch
params = {
    "decisiontreeclassifier__max_depth":[1, 2, 3, 4, 5, None]
}
```

```python
from sklearn.model_selection import GridSearchCV
```

```python
# Create an object of the GridSearchCV Class
grid = GridSearchCV(estimator=pipe, param_grid=params, cv=5, scoring="accuracy")

# Fit the training data
grid.fit(x_train, y_train)
```

```python
# Print the best parameters for the model
grid.best_params_
```

```python
# Print the overall accuracy
grid.best_score_
```

## **Export the Pipeline**

```python
import pickle
```

```python
pickle.dump(pipe, file=open("D:\Coding\Models\pipe.pkl", "wb"))
```

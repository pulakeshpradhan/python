[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/05_ML_without_Pipeline_1.ipynb)

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

# **Machine Learning without Pipeline - 1**

"Machine learning without pipelines" refers to the practice of implementing machine learning tasks and workflows without utilizing the structured concept of pipelines. In machine learning, pipelines are a systematic and organized approach for data preprocessing, feature engineering, model selection, training, and evaluation. However, there are situations where you may choose not to use pipelines, especially in simple or small-scale machine learning projects.


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
# Dropping the unnecessary columns
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
df.head()
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
x_train.head()
```

```python
y_train.head()
```

## **Data Preprocessing**

```python
# Check the information of the columns
df.info()
```

### **Apply SimpleImputer on 'Age' and 'Embarked' Columns**

```python
from sklearn.impute import SimpleImputer
```

```python
# Create an object of the SimpleImputer class
simple_imputer_age = SimpleImputer()
simple_imputer_embarked = SimpleImputer(strategy="most_frequent")

# Fit the training data
simple_imputer_age.fit(x_train[["Age"]])
simple_imputer_embarked.fit(x_train[["Embarked"]])

# Transform the 'Age' and 'Embarked' columns of the training data
x_train_age = simple_imputer_age.transform(x_train[["Age"]])
x_train_embarked = simple_imputer_embarked.transform(x_train[["Embarked"]])

# Transform the 'Age' and 'Embarked' columns of the testing data
x_test_age = simple_imputer_age.transform(x_test[["Age"]])
x_test_embarked = simple_imputer_embarked.transform(x_test[["Embarked"]])
```

```python
# Print the first 5 values of x_train_age
x_train_age[:5]
```

```python
# Print the first 5 values of x_train_embarked
x_train_embarked[:5]
```

### **Apply OneHot Encoder on 'Sex' and 'Embarked' Columns**

```python
from sklearn.preprocessing import OneHotEncoder
```

```python
# Create an object of the OneHotEncoder class
one_hot_encoder_sex = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
one_hot_encoder_embarked = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Fit the training data
one_hot_encoder_sex.fit(x_train[["Sex"]])
one_hot_encoder_embarked.fit(x_train_embarked)

# Transform the 'Sex' and 'Embarked' columns of the training data
x_train_sex = one_hot_encoder_sex.transform(x_train[["Sex"]])
x_train_embarked = one_hot_encoder_embarked.transform(x_train_embarked)

# Transform the 'Sex' and 'Embarked' columns of the testing data
x_test_sex = one_hot_encoder_sex.transform(x_test[["Sex"]])
x_test_embarked = one_hot_encoder_embarked.transform(x_test_embarked)
```

```python
# Print the first 5 values of x_train_sex
x_train_sex[:5]
```

```python
# Print the first 5 values of x_train_embarked
x_train_embarked[:5]
```

```python
# Drop the 'Age', 'Sex' and 'Embarked' column from the training data
x_train_remaining = x_train.drop(columns=["Age", "Sex", "Embarked"])
x_train_remaining.head()
```

```python
# Drop the 'Age', 'Sex' and 'Embarked' column from the testing data
x_test_remaining = x_test.drop(columns=["Age", "Sex", "Embarked"])
x_test_remaining.head()
```

```python
# Merge the processed columns with the reamaining dataframe
x_train_transformed = np.concatenate((x_train_remaining, x_train_age, x_train_sex, x_train_embarked), axis=1)
x_test_transformed = np.concatenate((x_test_remaining, x_test_age, x_test_sex, x_test_embarked), axis=1)
```

```python
# Print the x_train_transformed data
x_train_transformed
```

```python
one_hot_encoder_embarked.get_feature_names_out()
```

```python
# Assemble the column names of the transformed data
x_transformed_columns = np.array(x_train_remaining.columns)
x_transformed_columns = np.concatenate((x_transformed_columns, 
                                        simple_imputer_age.get_feature_names_out(),
                                        one_hot_encoder_sex.get_feature_names_out(),
                                        one_hot_encoder_embarked.get_feature_names_out()))
```

```python
x_transformed_columns
```

```python
# Convert the tranformed arrays into pandas dataframe
x_train_transformed = pd.DataFrame(x_train_transformed, columns=x_transformed_columns)
x_test_transformed = pd.DataFrame(x_test_transformed, columns=x_transformed_columns)
```

```python
x_train_transformed.head()
```

```python
x_test_transformed.head()
```

## **Build a DecisionTree Classifier**

```python
from sklearn.tree import DecisionTreeClassifier
```

```python
# Instantiate a DecisionTreeClassifier object
dt_classifier = DecisionTreeClassifier(random_state=0)

# Fit the training data
dt_classifier.fit(x_train_transformed, y_train)
```

## **Accuracy Assessment**

```python
# Predict the x_test_transformed data
y_pred = dt_classifier.predict(x_test_transformed)
y_pred
```

```python
from sklearn.metrics import accuracy_score
```

```python
# Print the overall accuracy of the decision tree model
accuracy_score(y_test, y_pred)
```

## **Export the Model**

```python
import pickle
```

```python
# Exporting the one_hot_encoder_sex
pickle.dump(one_hot_encoder_sex, file=open("D:\Coding\Models\ohe_sex.pkl", "wb"))

# Exporting the one_hot_encoder_embarked
pickle.dump(one_hot_encoder_embarked, file=open("D:\Coding\Models\ohe_embarked.pkl", "wb"))

# Exporting the decision tree classifier
pickle.dump(dt_classifier, file=open("D:\Coding\Models\decision_tree_model.pkl", "wb"))
```

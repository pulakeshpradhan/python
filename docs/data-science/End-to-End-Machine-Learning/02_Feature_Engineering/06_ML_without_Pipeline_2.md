[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/06_ML_without_Pipeline_2.ipynb)

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

# **Machine Learning without Pipeline - 2**


## **Import Required Libraries**

```python
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
```

## **Load the Model and Encoders**

```python
# Load the model and the encoders
ohe_sex = pickle.load(open("D:\Coding\Models\ohe_sex.pkl", "rb"))
ohe_embarked = pickle.load(open("D:\Coding\Models\ohe_embarked.pkl", "rb"))
clf = pickle.load(open("D:\Coding\Models\decision_tree_model.pkl", "rb"))
```

## **Take User Input**

```python
# Assume user input
# Pclass / Sex / Age / SibSp / Parch / Fare / Embarked
test_input = np.array([2, "male", 22.0, 1, 0, 25.0, "S"], dtype=object).reshape(1, 7)
```

```python
test_input
```

## **Transform the Input using Encoders**

```python
 # Encode the 'Sex'
test_input_sex = ohe_sex.transform(test_input[:, 1].reshape(1, 1))
test_input_sex
```

```python
# Encode the 'Embarked'
test_input_embarked = ohe_embarked.transform(test_input[:, -1].reshape(1, 1))
test_input_embarked
```

```python
# Extract the 'Age'
test_input_age = test_input[:, 2].reshape(1, 1)
test_input_age
```

```python
# Extract the remaining columns from test_input data
test_input_rem = test_input[:, [0, 3, 4, 5]]
test_input_rem
```

```python
# Merge the encoded data with the test input data
test_input_merged = np.concatenate((test_input_rem, test_input_age, test_input_sex, test_input_embarked), axis=1)
test_input_merged
```

```python
test_input_merged.shape
```

## **Predict the Input Data with the Model**

```python
predict = clf.predict(test_input_merged)
predict
```

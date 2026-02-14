[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/02_Feature_Engineering/08_ML_with_Pipeline_2.ipynb)

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

# **Machine Learning with Pipeline - 2**


## **Import Required Libraries**

```python
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
```

## **Load the Model** 

```python
pipe = pickle.load(open("D:\Coding\Models\pipe.pkl", "rb"))
```

## **Take User Input**

```python
# Assume user input
test_input = np.array([2, "male", 22.0, 1, 0, 25.0, "S"], dtype="object").reshape(1, 7)
```

## **Predict the Input Data with the Model**

```python
pipe.predict(test_input)
```

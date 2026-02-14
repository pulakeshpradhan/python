[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Image-Analysis-in-Remote-Sensing-with-Python/01_Images_Arrays_Matrices/04_Square_Matrices.ipynb)

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

# **Square Matrices**
**Author: Krishnagopal Halder**

Square matrices are a fundamental concept in linear algebra, characterized by having an equal number of rows and columns. In other words, a square matrix has the same number of rows as it does columns, resulting in a shape that resembles a square. The size or order of a square matrix is typically denoted by a single positive integer, such as "n" or "m," which represents both the number of rows and the number of columns.


## **01. Importing Required Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
import math
```

## **02. Elementary Properties**


### **01. Creation of Square Matrix**

```python
# Creating a two dimensional square matrix
A = np.mat([[1, 2], [3, 4]])
# Printing the matrix
A
```

```python
# Checking the dimension of the matrix A
A.ndim
```

### **02. Determinant of a Square Matrix**
![](https://i.stack.imgur.com/4LrnA.gif)

```python
# Calculating the determinant of 2x2 matrix A
det_A = np.linalg.det(A)
det_A
```

```python
# Visualization
plt.figure(figsize=(5, 5))
plt.quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=1, color="red", label="i (Unit Vector)")
plt.quiver(0, 0, 0, 1, angles="xy", scale_units="xy", scale=1, color="green", label="j (Unit Vector)")
plt.quiver(0, 0, 1, 2, angles="xy", scale_units="xy", scale=1, color="blue", label="Transformed i")
plt.quiver(0, 0, 3, 4, angles="xy", scale_units="xy", scale=1, color="orange", label="Transformed j")
plt.xlim([-4, 5])
plt.ylim([-4, 5])
plt.legend(loc="lower right")
plt.grid()
```

```python
# Calculating the determinant of a 3x3 matrix B
B = np.mat([[1, 0, 2], [4, 2, 1], [3, 5, 2]])
det_B = np.linalg.det(B)
det_B
```

### **03. Properties of the Determinant**

```python
# Creating another 2x2 square matrix B
B = np.mat([[2, 1], [3, 5]])
B
```

```python
# The determinant has the property, |AB| = |A||B|
AB = B * A
AB
```

```python
# Calculating the determinant of AB matrix
det_AB = np.linalg.det(AB)
det_AB
```

```python
# Calculating the determinant of B
det_B = np.linalg.det(B)
det_B
```

```python
# Proving the property, |AB| = |A||B|
round(det_AB) == round(det_A * det_B)
```

```python
# The determinant has the property, |A.T| = |A|
# Calculating the transpose of A
trans_A = A.T
trans_A
```

```python
# Proving the property, |A.T| = |A|
round(np.linalg.det(trans_A)) == round(np.linalg.det(A))
```

### **04. Identity Matrix**
<center><img src="https://images.deepai.org/glossary-terms/038fe0cc858f49179700a6973ea97dab/identity1.gif" style="max-width:450px; height:auto"></center>

```python
# Creating an identity matrix
identity = np.identity(2, dtype="int8")
identity
```

### **05. Properties of Identity Matrix**

```python
# For any A, IA = AI = A
IA = identity * A
IA
```

```python
AI = A * identity
AI
```

```python
# Proving the property
print("IA is AI:\n", IA == AI)
print("AI is A:\n", AI == A)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Image-Analysis-in-Remote-Sensing-with-Python/01_Images_Arrays_Matrices/03_Algebra_of_Vectors_and_Matrices.ipynb)

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

# **Algebra of Vectors and Matrices**
**Author: Krishnagopal Halder**


## **01. Importing Required Libraries**

```python
import numpy as np
import math
import matplotlib.pyplot as plt
```

## **02. Introduction to Vectors**
In linear algebra, vectors are fundamental objects used to represent quantities that have both magnitude and direction. A vector can be thought of as an ordered collection of numbers, known as components or coordinates, arranged in a specific order. Each component of a vector represents a coordinate along a particular dimension in a coordinate system. The number of components in a vector determines its dimensionality.

Vectors are commonly denoted by lowercase bold letters, such as **v** or **u**, or by lowercase letters with an arrow on top, such as ẋ or ẏ. For example, in a two-dimensional space, a vector **x** can be represented as **x** = [x₁, x₂], where x₁ and x₂ are the components of **x** along the x and y axes, respectively.


<center><img src="https://media5.datahacker.rs/2020/03/Picture36-1-1024x949.jpg" style="max-width:300px; height:auto"></center>


### **Elementary Properties**


#### **01. Vector Creation**

```python
# Creating a two dimensional vector
x = np.array([[1], [2]])
# Checking the dimension
print(x.ndim)
# Printing the x
x
```

#### **02. Sum of Two Column Vectors**

```python
# Creating another two dimensional vector
y = np.array([[3], [4]])
# Adding two vector x and y
z = x + y
```

```python
# Plotting all the vectors x, y, z
plt.figure(figsize=(6, 4))
plt.quiver(0, 0, x[0], x[1], angles="xy", scale_units="xy", scale=1, color="red", label="x")
plt.quiver(0, 0, y[0], y[1], angles="xy", scale_units="xy", scale=1, color="green", label="y")
plt.quiver(0, 0, z[0], z[1], angles="xy", scale_units="xy", scale=1, color="blue", label="z or (x+y)")
plt.xlim([0, 5])
plt.ylim([0, 7])
plt.grid()
plt.legend(loc="lower right")
plt.show()
```

#### **03. Transpose of a Column Vector**

```python
# Transpose of x
x.T
```

#### **04. Length or Euclidean Norm of a Vector**

```python
# Calculating length of vector x
len_x = np.linalg.norm(x)
print("Length of x:", round(len_x, 4))
```

#### **05. Inner Product or Dot Product of Two Vectors**
<center><img src="https://i.stack.imgur.com/yxY6H.gif" style="max-width:350px; height:auto"></center>

```python
# Calculating the dot product between x and y
dot_product = (x.T).dot(y)
dot_product
```

```python
x.T.dot(y)
```

```python
# Calculating the dot product by using angle between two vectors
# Creating a function to calculate angle between two vectors
def angle(v1, v2):
    len_v1 = np.linalg.norm(v1)
    len_v2 = np.linalg.norm(v2)
    dot_product = (v1.T).dot(v2)
    arc_cos = math.acos(dot_product / (len_v1 * len_v2)) # angle in radian
    return arc_cos

# Deriving the angle between x and y
angle = angle(x, y)
print("Angle between x and y:", round(angle, 4))
print("Angle between x and y in Degree:", round(angle * (180 / math.pi), 4))
```

```python
# Calculating dot product
dot_product = np.linalg.norm(x) * np.linalg.norm(y) * math.cos(angle)
dot_product
```

#### **06. Vector Representation in terms of Orthogonal Unit Vectors**
<center><img src="https://qph.cf2.quoracdn.net/main-qimg-9d1a2481c20bf8720bae101817dc6e4b" style="max-width:650x; height: auto"></center>

```python
# Creating two unit vectors
i = np.array([[1], [0]])
j = np.array([[0], [1]])
# Representing vector x in terms of two unit vectors, i and j, respectively
x_new = x[0] * i + x[1] * j
x_new
```

## **03. Introduction to Matrices**
Matrices are a fundamental concept in linear algebra and are used to represent and manipulate linear systems of equations. In linear algebra, a matrix is a rectangular array of numbers or symbols arranged in rows and columns. The size of a matrix is specified by its dimensions, which are given by the number of rows and columns. For example, a matrix with m rows and n columns is said to have dimensions m x n.

Each entry in a matrix is called an element and is identified by its position in the matrix using the row and column indices. The row index is typically denoted by i, and the column index is denoted by j. Thus, the element in the i-th row and j-th column of a matrix A is written as A[i, j].

<center><img src="http://www.learnattic.com/wp-content/uploads/2019/10/Matrices-concept-representation.jpg" style="max-width:550px; height:auto"></center>


### **Elementary Properties**


#### **01. Matrix Creation**

```python
# Creating a 2x2 matrix
A = np.mat([[3, 2], [1, 4]])
A
```

#### **02. Multiplication of a Matrix with a Vector**

```python
# Multiplying matrix A with vector x
matrix_mult_vec = A * x
matrix_mult_vec
```

```python
# Plotting the linear transformation of vector x after multiplying with matrix A
plt.quiver(0, 0, x[0], x[1], 
           angles="xy", scale_units="xy", scale=1, color="red", label="Original Vector")
plt.quiver(0, 0, matrix_mult_vec[0], matrix_mult_vec[1], 
           angles="xy", scale_units="xy", scale=1, color="green", label="Tranformed Vector")
plt.xlim([0, 8])
plt.ylim([0, 10])
plt.grid()
plt.legend(loc="lower right")
plt.show()
```

#### **03. Product or Multiplication of Two Matrices**
The product of two matrices is allowed only when the first matrix has the same number of columns as the second matrix has rows.

```python
# Creating another matrix B
B = np.mat([[4, 2], [1, 3]])
# Printing two matrices
print("Matrix A:\n", A)
print("Matrix B:\n", B)
# Calculating the product of two 2x2 matrices
print("Product of A and B:\n", A * B)
```

#### **04. Properties of Matrix Multiplication**

```python
# Matrix multiplication is not commutative, AB is not equal to BA
(A * B) is (B * A)
```

```python
# Matrix multiplication is associative, (AB)C = A(BC) or (AB)C = A(BC) = ABC
# Creating another matrix C
C = np.mat([[2, 3], [4, 1]])
C
```

```python
# Multiplying A and B
A_B = B * A
# Multiplying B and C
B_C = C * B
```

```python
# Checking the associativity
(C * A_B) == (B_C * A)
```

```python
(C * B * A) == (B_C * A)
```

#### **05. Outer Product of Two Vectors**

```python
# Outer product of vectors x and y
x * y.T
```

#### **06. Transpose of a Matrix**

```python
# Creating a 3x3 matrix
D = np.mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
D
```

```python
# Transpose of matrix D
D.T
```

#### **07. Properties of Transpose**

```python
(A + B).T == (A.T) + (B.T)
```

```python
(A * B).T == (B.T) * (A.T)
```

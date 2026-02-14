# Numpy Attributes and Methods
NumPy is a powerful library for working with arrays and matrices in Python. It provides various attributes and methods that can be used to manipulate and analyze arrays. In this section, we will discuss some of the important NumPy attributes and methods.


```python
import numpy as np
```


```python
# Creating a NumPy 2-D array
myArr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```

## 01. NumPy Attributes

### 01. 'T' Attribute
In NumPy, the T attribute is used to get the transpose of a matrix. The transpose of a matrix is obtained by interchanging the rows and columns of the matrix.


```python
myArr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
# Creating transpose of the matrix
myArr_transpose = myArr.T
myArr_transpose
```




    array([[1, 4, 7],
           [2, 5, 8],
           [3, 6, 9]])



### 02. 'flat' Attribute
In NumPy, the flat attribute is used to get a 1-dimensional iterator over a multi-dimensional array. The flat attribute returns a flat iterator that traverses the array in row-major (C-style) order, which means that it first traverses the rows of the array, and then the columns.


```python
myArr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
myArr.flat
```




    <numpy.flatiter at 0x160f8c39a20>




```python
for item in myArr.flat:
    print(item)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    

### 03. 'ndim' Attribute
This attribute returns the number of dimensions of the array.


```python
myArr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
myArr.ndim
```




    2



### 04. 'size' Attribute
This attribute returns the total number of elements in the array.


```python
myArr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
myArr.size
```




    9



### 05. 'nbytes' Attribute
In NumPy, the nbytes attribute is used to get the total number of bytes occupied by the array data in memory. 


```python
myArr.nbytes
```




    36



## 02. NumPy Methods


```python
# Creating a NumPy 1-D array
myArr2 = np.array([45, 48, 25, 87, 16])
```

### 01. 'argmax' Method
np.argmax(arr, axis=None, out=None) returns the indices of the maximum values along an axis.


```python
# argmax in 1-D array
myArr2.argmax()
```




    3




```python
# argmax in 2-D array
myArr.argmax()
```




    8




```python
# Finding maximum values along axis
myArr.argmax(axis=0)
```




    array([2, 2, 2], dtype=int64)



### 02. 'argmin' Method
np.argmin(arr, axis=None, out=None) returns the indices of the minimum values along an axis.


```python
# argmin in 1-D array
myArr2.argmin()
```




    4




```python
# argmin in 2-D array
myArr.argmin()
```




    0




```python
# Finding minimum values along axis
myArr.argmin(axis=0)
```




    array([0, 0, 0], dtype=int64)



### 03. 'argsort' Method
argsort() is a method provided by NumPy that returns the indices that would sort an array in ascending or descending order.


```python
# argsort in 1-D array
myArr2.argsort()
```




    array([4, 2, 0, 1, 3], dtype=int64)




```python
# argsort in 2-D array
myArr.argsort()
```




    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]], dtype=int64)




```python
# Sorting values along axis
myArr.argsort(axis=0)
```




    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2]], dtype=int64)



### 04. 'ravel' Method
ravel() is used to flatten an array into a 1D array. 


```python
myArr.ravel()
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])



### 05. 'reshape' Method
reshape() is used to change the shape of an array.


```python
myArr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
myArr.reshape((9, 1))
```




    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6],
           [7],
           [8],
           [9]])



# Introduction to NumPy 

NumPy is a library for the Python programming language that provides support for arrays, matrices, and other numerical operations. It is an essential library for scientific computing in Python. In this module, you will learn about the basics of NumPy and how to use it for numerical computations.

**Prerequisites:**
Before starting with NumPy, it is recommended to have a basic understanding of Python programming language, especially control statements, functions, and data types.

<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/1280px-NumPy_logo_2020.svg.png" style = "max-width: 350px; height: auto"></center>

## 01. Installation and Importing
To install NumPy, use the pip package manager in the terminal by typing the following command:


```python
# !pip install numpy
```


```python
import numpy as np
```

## 02. Creation of One-Dimensional Arrays
In NumPy, you can create a one-dimensional array using the numpy.array() method.


```python
myArr = np.array([1, 2, 3, 4, 5], dtype="int8")
myArr
```




    array([1, 2, 3, 4, 5], dtype=int8)




```python
# Accessing elements of an one-dimensional array
print(myArr[0], myArr[3])
```

    1 4
    


```python
# Print the data type
myArr.dtype
```




    dtype('int8')



## 03. Creation of Two-Dimensional Arrays
In NumPy, you can create a two-dimensional array using the numpy.array() method and passing a list of lists as an argument.


```python
myArr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="int8")
myArr2
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]], dtype=int8)




```python
# Accessing elements of two-dimensional array
print(myArr2[0])
print(myArr2[2][0])
print(myArr2[0, 2])
```

    [1 2 3]
    7
    3
    


```python
# Print the data type
myArr2.dtype
```




    dtype('int8')




```python
# Print the shape of the array
myArr2.shape
```




    (3, 3)




```python
# Changing the element of a two-dimensional array
myArr2[0, 2] = 10
myArr2
```




    array([[ 1,  2, 10],
           [ 4,  5,  6],
           [ 7,  8,  9]], dtype=int8)



## 04. Other Ways of Array Creation

### 01. Array Creation from Other Python Structures


```python
# Array creation from List
myList = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
listArray = np.array(myList)
listArray
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
listArray.dtype
```




    dtype('int32')




```python
listArray.shape
```




    (3, 3)




```python
# Array creation from tuple
myTuple = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
tupleArray = np.array(myTuple)
tupleArray
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
tupleArray.dtype
```




    dtype('int32')




```python
# Array cration from set
mySet = {1, 2, 3, 4, 5, 6}
setArray = np.array(mySet)
setArray
```




    array({1, 2, 3, 4, 5, 6}, dtype=object)




```python
# dtype object is not efficient for numeric calculations
setArray.dtype
```




    dtype('O')



### 02. Intrinsic NumPy Array Creation Objects


```python
# Array creation using zeros function
zeros = np.zeros((3, 3))
zeros
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
# Array creation using range function
rngArray = np.arange(0, 11, 2)
rngArray
```




    array([ 0,  2,  4,  6,  8, 10])




```python
# Array creation using linspace function
# linspace function is used to create a linearly spaced array
lspace = np.linspace(start=0, stop=50, num=5)
lspace
```




    array([ 0. , 12.5, 25. , 37.5, 50. ])




```python
lspace2 = np.linspace(1, 2, 5)
lspace2
```




    array([1.  , 1.25, 1.5 , 1.75, 2.  ])




```python
# Array creation using empty function
# empty function is used to create an empty array
empArray = np.empty((4, 6)) # Elements will be random in this case
empArray
```




    array([[6.23042070e-307, 4.67296746e-307, 1.69121096e-306,
            2.78148153e-307, 1.29060531e-306, 8.45599366e-307],
           [7.56593017e-307, 1.33511290e-306, 1.02359645e-306,
            1.24610383e-306, 1.69118108e-306, 8.06632139e-308],
           [1.20160711e-306, 1.69119330e-306, 1.29062229e-306,
            1.60217812e-306, 1.37961370e-306, 1.69118515e-306],
           [1.11258277e-307, 1.05700515e-307, 1.11261774e-306,
            1.29060871e-306, 8.34424766e-308, 2.12203497e-312]])




```python
# Array creation using empty_like function
# empty_like function is used to generate a copy of previously created array
empArray2 = np.empty_like(lspace)
empArray2
```




    array([1.  , 1.25, 1.5 , 1.75, 2.  ])




```python
# Array creation using identity function
# identity function is used to create an identity matrix
identityMatrix = np.identity(4)
identityMatrix
```




    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])




```python
identityMatrix.shape
```




    (4, 4)



## 05. Reshaping NumPy 1-D Array to 2-D Array


```python
# Creating NumPy one-dimensional array using range function
arr = np.arange(50)
arr
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])




```python
# Reshaping 1-D array to 2-D array using range function
arr = arr.reshape((5, 10))
arr
```




    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
           [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])




```python
arr.shape
```




    (5, 10)



## 06. Converting NumPy 2-D Array to 1-D Array


```python
# Two-dimensional array
arr
```




    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
           [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])




```python
# Converting 2-D array into 1-D array
arr = arr.ravel()
arr
arr
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])



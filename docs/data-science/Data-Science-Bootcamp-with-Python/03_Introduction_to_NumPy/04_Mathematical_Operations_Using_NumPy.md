[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/03_Introduction_to_NumPy/04_Mathematical_Operations_Using_NumPy.ipynb)

# Mathematical Operations Using NumPy
NumPy provides a wide range of mathematical operations that can be performed on arrays. These operations include basic arithmetic operations (addition, subtraction, multiplication, division), as well as more advanced mathematical functions (trigonometric functions, logarithmic functions, etc.).


```python
import numpy as np
```


```python
# Creating our first array
arr1 = np.array([[1, 3, 5], [4, 7, 4], [3, 6, 1]])
arr1
```




    array([[1, 3, 5],
           [4, 7, 4],
           [3, 6, 1]])




```python
# Creating our second array
arr2 = np.array([[2, 7, 6], [3, 4, 8], [1, 7, 3]])
arr2
```




    array([[2, 7, 6],
           [3, 4, 8],
           [1, 7, 3]])



## 01. Basic Matrix Operations
A wide range of matrix operations can be performed on arrays by using NumPy. These operations include basic arithmetic operations (addition, subtraction, multiplication, division), as well as more advanced matrix operations (determinants, inverses, eigenvalues, etc.).

### 01. Matrix Addition


```python
# Addition
arr1 + arr2
```




    array([[ 3, 10, 11],
           [ 7, 11, 12],
           [ 4, 13,  4]])



### 02. Matrix Subtraction


```python
# Subtraction
arr1 - arr2
```




    array([[-1, -4, -1],
           [ 1,  3, -4],
           [ 2, -1, -2]])



### 03. Matrix Multiplication


```python
# Multiplication
arr1 * arr2
```




    array([[ 2, 21, 30],
           [12, 28, 32],
           [ 3, 42,  3]])



### 04. Matrix Division


```python
# Division
arr1 / arr2
```




    array([[0.5       , 0.42857143, 0.83333333],
           [1.33333333, 1.75      , 0.5       ],
           [3.        , 0.85714286, 0.33333333]])



## 02. Basic Statistical Operations
NumPy provides several methods to perform basic statistical operations on arrays such as sqrt(), sum(), min(), and max().

### 01. 'sqrt' Method
The numpy.sqrt() method is used to calculate the square root of each element in a NumPy array.


```python
np.sqrt(arr1)
```




    array([[1.        , 1.73205081, 2.23606798],
           [2.        , 2.64575131, 2.        ],
           [1.73205081, 2.44948974, 1.        ]])



### 02. 'sum' Method
The sum() method returns the sum of all elements in the array or along a specified axis.


```python
np.sum(arr1)
```




    34



### 03. 'min' Method
The min() method returns the minimum value in the array or along a specified axis.


```python
np.min(arr1, axis=0)
```




    array([1, 3, 1])




```python
# Return minimum value of an array
arr1.min()
```




    1



### 04. 'max' Method
The max() method returns the maximum value in the array or along a specified axis.


```python
np.max(arr1, axis=0)
```




    array([4, 7, 5])




```python
# Return maximum value of an array
arr1.max()
```




    7



## 03. Other Useful Methods
NumPy provides several methods to work with Boolean arrays, including where(), count_nonzero(), and nonzero(). 

### 01. 'where' Method
The where() method returns an array of the same shape as the input array, where each element is replaced by either the value x if the corresponding element in the Boolean mask is True, or the value y if the corresponding element in the Boolean mask is False. You can also use the where() method to extract the indices of the elements that meet a certain condition.


```python
arr1
```




    array([[1, 3, 5],
           [4, 7, 4],
           [3, 6, 1]])




```python
np.where(arr1>5, 1, 0)
```




    array([[0, 0, 0],
           [0, 1, 0],
           [0, 1, 0]])



### 02. 'count_nonzero' Method
The count_nonzero() method returns the number of non-zero elements in the input array or along a specified axis.


```python
np.count_nonzero(arr1)
```




    9




```python
# Changing the element of arr1 to 0 at the index position off [1, 2]
arr1[1, 2] = 0
arr1
```




    array([[1, 3, 5],
           [4, 7, 0],
           [3, 6, 1]])




```python
np.count_nonzero(arr1)
```




    8



### 03. 'nonzero' Method
The nonzero() method returns a tuple of arrays, one for each dimension of the input array, containing the indices of the non-zero elements in that dimension. You can also use the nonzero() method to extract the non-zero elements of an array.


```python
arr1
```




    array([[1, 3, 5],
           [4, 7, 0],
           [3, 6, 1]])




```python
arr1[1, 2]
```




    0




```python
np.nonzero(arr1)
```




    (array([0, 0, 0, 1, 1, 2, 2, 2], dtype=int64),
     array([0, 1, 2, 0, 1, 0, 1, 2], dtype=int64))



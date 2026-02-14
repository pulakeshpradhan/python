```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
myList = [1, 2, 3, 4]
myList
```




    [1, 2, 3, 4]




```python
myList2 = [[1,2], [3,4]]
myList2
```




    [[1, 2], [3, 4]]




```python
myArr1 = np.array([1, 2, 3, 4])
myArr1
```




    array([1, 2, 3, 4])




```python
myArr1[3]
```




    4




```python
type(myArr1)
```




    numpy.ndarray




```python
myArr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="int64")
myArr2
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]], dtype=int64)




```python
myArr2[2, 1]
```




    8




```python
myArr2.dtype
```




    dtype('int64')




```python
myArr2.nbytes
```




    72




```python
myArr2.shape
```




    (3, 3)




```python
tup = ((1,2), (3, 4))
myArr3 = np.array(tup)
myArr3
```




    array([[1, 2],
           [3, 4]])




```python
arr4 = np.ones((3, 3))
arr4
```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])




```python
arr5 = np.zeros((3, 3))
arr5
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
arr6 = np.arange(1, 101, 2)
arr6
```




    array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33,
           35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67,
           69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99])




```python
arr7 = np.linspace(1, 10, 5)
arr7
```




    array([ 1.  ,  3.25,  5.5 ,  7.75, 10.  ])




```python
arr8 = np.empty((3, 3))
arr8
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
identityMatrix = np.identity(6, dtype="int8")
identityMatrix
```




    array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1]], dtype=int8)




```python
identityMatrix.reshape((4, 9))
```




    array([[1, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 1]], dtype=int8)




```python
identityMatrix.ravel()
```




    array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=int8)



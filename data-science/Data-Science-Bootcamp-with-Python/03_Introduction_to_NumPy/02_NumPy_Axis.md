# Numpy Axis
In NumPy, the axis parameter is used to specify the dimension of an array along which a particular operation should be performed. It is an important parameter in many NumPy functions that deal with multi-dimensional arrays. Understanding how to use the axis parameter is crucial in performing complex operations on multi-dimensional arrays.

The axis parameter can take values of 0, 1, 2, and so on, where 0 represents the first dimension (rows), 1 represents the second dimension (columns), and so on.

<center><img src="https://i.imgur.com/mg8O3kd.png" style="max-width:1200px; height: auto"></center>


```python
import numpy as np
```


```python
# Creating a 2-D list
myList = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```


```python
# Creating a 2-D NumPy array from the list
myArr = np.array(myList)
myArr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])



## 01. Summing Elements along a Specific Axis


```python
# Summing elements along axis 0 (Row)
row_sum = myArr.sum(axis=0)
row_sum
```




    array([12, 15, 18])




```python
# Summing elements along axis 1 (Column)
column_sum = myArr.sum(axis=1)
column_sum
```




    array([ 6, 15, 24])



## 02. Finding the Maximum Element along a Specific Axis


```python
myArr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
# Finding the maximum element along axis 0 (Row)
max_axis_0 = myArr.max(axis=0)
max_axis_0
```




    array([7, 8, 9])




```python
# Finding the maximum element along axis 1 (Column)
max_axis_1 = myArr.max(axis=1)
max_axis_1
```




    array([3, 6, 9])



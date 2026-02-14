[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/03_Introduction_to_NumPy/05_Advantages_of_Using_NumPy.ipynb)

# Advantages of Using NumPy
NumPy arrays are more memory-efficient than Python lists because they are homogeneous, which means that all elements in the array are of the same data type. This allows NumPy to store the data more efficiently in memory.

When you create a Python list, Python has to allocate memory for each element of the list, as well as for the list object itself. This means that lists can use more memory than necessary if the elements have different data types.

In contrast, NumPy arrays are designed to be memory-efficient. When you create a NumPy array, NumPy allocates a block of memory for the entire array, based on the data type and size of the array. This means that NumPy arrays can use less memory than equivalent lists, especially for large datasets.


```python
import sys
import numpy as np
```


```python
# Creating a list
myList = [1, 2, 3, 4, 5]
myList
```




    [1, 2, 3, 4, 5]




```python
# Creating a NumPy array from the list
myArr = np.array(myList)
myArr
```




    array([1, 2, 3, 4, 5])




```python
# Print the size of the python list
# Memory in Bytes
sys.getsizeof(myList) * len(myList)
```




    600




```python
# Print the size of the NumPy array
# Memory in Bytes
myArr.itemsize * myArr.size
```




    20



## Convert NumPy Array to List


```python
arr_to_lst = myArr.tolist()
arr_to_lst
```




    [1, 2, 3, 4, 5]




```python
sys.getsizeof(arr_to_lst)
```




    96



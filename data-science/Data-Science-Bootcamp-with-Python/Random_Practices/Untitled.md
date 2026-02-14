```python
import numpy as np
```


```python
myArr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
myArr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
myArr.sum(axis=0)
```




    array([12, 15, 18])




```python
myArr.max(axis=0)
```




    array([7, 8, 9])




```python
myArr.T
```




    array([[1, 4, 7],
           [2, 5, 8],
           [3, 6, 9]])




```python
myArr.flat
```




    <numpy.flatiter at 0x1fda9706e40>




```python
print(myArr)
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    

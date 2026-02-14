[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Two Dimensional List.ipynb)

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

# Two-Dimensional(2-D) List

```python
myList = [1, 2, 3, 4]
```

```python
student1 = [9, 8, 6, 5]
student2 = [8, 10, 4, 7]
student3 = [7, 3, 8, 9]
student4 = [10, 8, 7, 9]
```

```python
marks_of_students = [student1, student2, student3, student4]
```

```python
marks_of_students
```

```python
import numpy as np
```

```python
myArr = np.array([1, 2, 3, 4])
```

```python
myArr.ndim
```

```python
myArr2D = np.array([[1, 2, 3, 4], [5, 6, 7, 9], [10, 11, 12, 13]])
```

```python
myArr2D
```

```python
myArr2D.shape
```

```python
myArr2D.ndim
```

## Accessing Values in a Two-Dimensional List

```python
marks_of_students
```

```python
# List at index 0 in marks_of_students
marks_of_students[2]
```

```python
# Element at index 1 in list at index 1
marks_of_students[1][1]
```

```python
# Element at index 3 in list at index 1
marks_of_students[3][1]
```

```python
myList = [1, 2, 3]
```

```python
myList[2]
```

## Input of Two-Dimensional List


### Line Separated Input

```python
# Number of rows
print("How many number of rows do you want?")
n = int(input())
inputList = [[int(col) for col in input().split()] for row in range(n)]
```

```python
inputList
```

### Space Separated Input

```python
# Number of rows
print("How many number of rows do you want: ")
row = int(input())

# Number of columns
print("How many number of columns do you want: ")
col = int(input())

# Converting input string to list
inputList = input().split()

finalList = [[int(inputList[i*col+j]) for j in range(col)] for i in range(row)]
print(finalList)
```

## Printing in Multiple Lines Like a Matrix

```python
finalList
```

```python
# Iterate in row of 2-D list
for row in finalList:
    for col in row:
        print(col, end=" ")
    print()
```

## Jagged List

```python
jg_list = [[1, 2, 3], 4, 5, [5, 6, 7,9], [2, 0]]
```

```python
jg_list[3]
```

```python
jg_list[3][3]
```

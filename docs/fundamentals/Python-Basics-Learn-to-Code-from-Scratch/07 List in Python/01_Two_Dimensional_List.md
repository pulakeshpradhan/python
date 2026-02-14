[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/07 List in Python/01_Two_Dimensional_List.ipynb)

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

## Two-Dimensional(2-D) List


A two-dimensional list is a list of lists where each list within the main list represents a row of data. Two-dimensional lists are commonly used in data analysis and scientific computing to represent matrices and tables.


### Creating a Two-Dimensional List
Consider an example of recording test scores of 4 students, in 4 subjects. Such data can be represented as a two-dimensional list

```python
student1 = [9, 8, 6, 5]
student2 = [8, 10, 4, 7]
student3 = [7, 3, 8, 9]
student4 = [10, 8, 7, 9]
```

```python
marks_of_students = [student1, student2, student3, student4]
print(marks_of_students)
```

### Accessing Values in a Two-Dimensional List
To access an element in a two-dimensional list, we use the row and column indices of the element. The row index specifies the inner list, and the column index specifies the position within the inner list.

```python
# List at index 0 in marks_of_students
print(marks_of_students[0])

# Element at index 2 in list at index 0
print(marks_of_students[0][2])
```

### Input of Two-Dimensional List
We can also input data into a two-dimensional list from user input using loops and the input function. We will discuss two common ways of taking user input:
* Line Separated Input: Different rows in different lines.
* Space Separated Input: Taking input in a single line.


#### Line Separated Input

```python
# Number of rows
n = int(input())
inputList = [[int(col) for col in input().split()] for row in range(n)]
print(inputList)
```

#### Space Separated Input

```python
# Number of rows
row = int(input())

# Number of columns
col = int(input())

# Converting input string to list
inputList = input().split(" ") 

finalList = [[int(inputList[i*col+j])for j in range(col)]for i in range(row)]

print(finalList)
```

### Printing in Multiple Lines Like a Matrix
We can use a nested for loops to print a 2-D List. The outer loop iterates over the rows and the inner loop will itearte over columns.

```python
myList = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

# Iterate in row of 2-D List
for row in myList:
    for col in row:
        print(col, end=" ")
    print()
```

### Concept of Jagged Lists
In Python, a jagged list is a list of lists where the sublists may have different lengths. Jagged lists are useful when we need to store data that is not necessarily uniform or when we want to conserve memory by not allocating space for unused elements. Jagged lists are commonly used in data science, natural language processing, and machine learning.

```python
jgList1 = [[1, 2, 3], 4, 5, [6, 7], [8, 9, 10]]
jgList2 = [[10, 11, 12], [18, 19], [20, 22, 23, 24], [10, 12]]
```

```python
print(jgList1[3])
print(jgList2[2])
```

```python
a = [1, 2, 3, 4]
print(a.index(2))
```

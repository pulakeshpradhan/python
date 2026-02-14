[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Function_in_Python.ipynb)

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

# Function in Python

```python
def addTwoNumbers(x, y):
    "This function returns the sum of two numbers"
    sum_of_two_numbers = x + y
    return sum_of_two_numbers
```

```python
addTwoNumbers(15, 45)
```

```python
def multiplyTwoNumbers(x, y):
    "This function returns the product of two numbers"
    product = x * y
    return product
```

```python
multiplyTwoNumbers(1526, 152)
```

```python
def divideTwoNumbers(x, y):
    "This function returns the division of two numbers"
    divide = x / y
    return divide
```

```python
divideTwoNumbers(10, 2)
```

## Create a function that will calculate the standard deviation of a dataset

```python
myLst = [10, 14, 36, 12, 14, 75, 28, 45, 31, 98, 18, 24, 45, 30]
```

```python
sum(myLst) # Sum of all the numbers
```

```python
len(myLst) # Number of items
```

```python
# Mean = (sum of all the numbers) / (no of items)
mean = sum(myLst) / len(myLst)
mean
```

```python
from math import sqrt
def standard_deviation(lst):
    "This function return the standard deviation value of a dataset"
    mean = sum(lst) / len(lst)
    sum_of_squares = 0
    
    for num in lst:
        deviation = num - mean
        sqrd_deviation = deviation ** 2
        sum_of_squares += sqrd_deviation
        
    std_deviation =  sqrt((sum_of_squares / (len(lst)-1)))
    return std_deviation
```

```python
standard_deviation(myLst)
```

```python
standard_deviation([1, 2, 3, 4, 5])
```

```python
def average(lst):
    "This function return the average of a dataset"
    sum_of_dataset = sum(lst)
    no_of_items = len(lst)
    avg = sum_of_dataset / no_of_items
    return avg
```

```python
myData = [1, 2, 3, 4, 5]
```

```python
average(myData)
```

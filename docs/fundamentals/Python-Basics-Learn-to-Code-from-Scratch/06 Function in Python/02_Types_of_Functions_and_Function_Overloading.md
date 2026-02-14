[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/06 Function in Python/02_Types_of_Functions_and_Function_Overloading.ipynb)

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

## Types of Functions


In Python, there are several types of functions that serve different purposes. Here are some of the most commonly used types:
* Built-in Functions: These are functions that are pre-defined in Python and can be used directly in our code without the need for any additional definitions. Examples include print(), len(), and input().

* User-Defined Functions: These are functions that are defined by the user for a specific purpose. They can be called from anywhere in the code and can take one or more arguments. User-defined functions can be very simple or very complex, depending on the task they are designed to perform.

* Anonymous Functions (Lambda Functions): These are small, one-line functions that can be defined without a name using the "lambda" keyword. They are commonly used when you need to pass a function as an argument to another function.


### Built-in Functions

```python
name = "GeoNext"
# Built-in functions
print(name)
print(len(name))
```

### User-Defined Functions

```python
def multiply(a, b):
    """This function returns the multiplication of two numbers"""
    return a * b

multiply(2, 3)
```

### Lambda Function

```python
square = lambda x: x**2
square(2)
```

```python
divide = lambda x, y: x / y
divide(6, 2)
```

## Function Overloading


Function overloading is a programming concept that allows a function to have multiple implementations with the same name but different parameters. In some programming languages such as C++ and Java, function overloading is supported natively. However, in Python, function overloading is not supported in the same way, but there are ways to achieve similar functionality.

In python, we can define multiple functions with the same name but only the last function will be considered. all the rest gets hidden.

```python
def add(x, y):
    return x + y

def add(x, y, z):
    return x + y + z
```

```python
# add(1, 2) # Throws an error
add(1, 2, 3)
```

In the above code, there are two add() methods, where only last methed of them can be used. Calling any of other methods will produce an error. Like here calling add(1, 2) will throw an error.

This issue can be overcomed by the following method:

```python
# !pip install multipledispatch
```

```python
from multipledispatch import dispatch
```

```python
@dispatch(int, int)
def add(x, y):
    return x + y

@dispatch(int, int, int)
def add(x, y, z):
    return x + y + z
```

```python
add(1, 2)
```

```python
add(1, 2, 3)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/10 Exception Handling/03_try_except_else_Statement.ipynb)

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

## try-except-else Statement
We can also use the else statement with the try-except statement in which, we can place the code which will be executed in the scenario if no exception occurs in the else block. The syntax is given below:


<center><img src="https://files.realpython.com/media/try_except_else.703aaeeb63d3.png" style="max-width: 650px; height: auto"></center>

<!-- #region -->
**Syntax:**
```python
try:
    # Some code...
    
except SomeException as e:
    # Optional block
    # Handling of exception (if required)
    
else:
    # execute if no exception
```
<!-- #endregion -->

**Example:**

```python
def divide_numbers(x, y):
    try:
        result = x / y
        
    except ZeroDivisionError as e:
        print("Error: Divison by zero is not allowed.")
    
    else:
        print("Result:", result)
```

```python
divide_numbers(10, 0)
```

```python
divide_numbers(10, 2)
```

We get this output because there is no exception in the try block when x is 10 and y is 2, and hence the else block is executed. If there was an exception in the try block, in the case of x=10, y=0, the else block will be skipped and except block will be executed.

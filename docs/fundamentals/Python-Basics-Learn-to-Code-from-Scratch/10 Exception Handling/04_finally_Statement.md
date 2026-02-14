[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/10 Exception Handling/04_finally_Statement.ipynb)

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

## finally Statement
The try statement in Python can have an optional **finally** clause. This clause is executed no matter what and is generally used to release external resources. Here is an example of file operations to illustrate this: 

Let’s first understand how the try and except works - 
* First, the try clause is executed i.e. the code between try and except clause. 
* If there is no exception, then only try clause will run, except clause will not get executed. 
* If any exception occurs, the try clause will be skipped and except clause will run. 
* If any exception occurs, but the except clause within the code doesn’t handle it, it is passed on to the outer try statements. If the exception is left unhandled, then the execution stops. 
* A try statement can have more than one except clause.


<center><img src="https://files.realpython.com/media/try_except_else_finally.a7fac6c36c55.png" style="max-width: 650px; height:auto"></center>

<!-- #region -->
**Syntax:**
```python
try:
    # Some code...
    
except:
    # Optional block
    # Handling of exception (if required)
    
else:
    # Execute if no exception
    
finally:
    # Some code... (always executed)
```
<!-- #endregion -->

**Example:**

```python
def divide(x, y):
    try:
        result = x / y
        
    except ZeroDivisionError:
        print("Error! Division by zero is not allowed.")
        
    else:
        print("Result:", result)
        
    finally:
        # This block is always executed
        # regardless of exception generation
        print("This is always executed.")
```

```python
# Look at parameters and note the working of program
divide(5, 2)
```

```python
divide(5, 0)
```

### Raising Exception in Python
In Python programming, exceptions are raised when errors occur at runtime. We can also manually raise exceptions using the raise keyword. We can optionally pass values to the exception to clarify why that exception was raised. Given below are some examples to help you understand this better.

<!-- #region -->
```python
>>> raise KeyboardInterrupt
Traceback (most recent call last):
...
KeyboardInterrupt
```
<!-- #endregion -->

<!-- #region -->
```python
>>> raise MemoryError("This is an argument")
Traceback (most recent call last):
...
MemoryError: This is an argument
```
<!-- #endregion -->

**Example:**

```python
try:
    a = int(input("Enter a number: "))
    
    if a <= 0:
        raise ValueError("Please enter a positive number.")
        
except ValueError as ve:
    print(ve)
```

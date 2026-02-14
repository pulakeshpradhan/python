[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/10 Exception Handling/02_try_except_Statement.ipynb)

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

## try-except Statement
In Python, exceptions can be handled using try-except blocks. 
* If the Python program contains suspicious code that may throw the exception, we must place that code in the try block. 
* The try block must be followed by the except statement, which contains a block of code that will be executed in case there is some exception in the try block. 
* We can thus choose what operations to perform once we have caught the exception


<center><img src="https://files.realpython.com/media/try_except.c94eabed2c59.png" style="max-width:650px; height:auto"></center>

<!-- #region -->
**Syntax:**
```python
try:
    # Some Code...
    
except SomeException as e:
    # Optional block
    # Handling of exception (if required)
```

**Example:**
<!-- #endregion -->

```python
myList = ["a", 0, 2]

for i in myList:
    try:
        print("The list item is:", i)
        reciprocal = 1/int(i)
        print("The reciprocal of", i, "is", reciprocal)
        
    except Exception as e: #Using Exception class
        print("Oops!", e.__class__, "occurred.")
        print()
```

* In this program, we loop through the values of a list 'myList'. 
* As previously mentioned, the portion that can cause an exception is placed inside the try block. 
* If no exception occurs, the except block is skipped and normal flow continues (for last value). 
* But if any exception occurs, it is caught by the except block (first and second values).
* Here, we print the name of the exception using the exc_info() function inside sys module. 
* We can see that element “a” causes ValueError and 0 causes ZeroDivisionError.


### Catching Specific Exceptions in Python
* In the above example, we did not mention any specific exception in the **except** clause. 
* This is not a good programming practice as it will catch all exceptions and handle every case in the same way. 
* We can specify which exceptions an **except** clause should catch. 
* A try clause can have any number of **except** clauses to handle different exceptions, however, only one will be executed in case an exception occurs. 
* You can use multiple **except** blocks for different types of exceptions. 
* We can even use a tuple of values to specify multiple exceptions in an **except** clause. Here is an example to understand this better:


**Example:**

```python
try:
    a = 10/0
    print(a)
    
except (ArithmeticError, IOError):
    print("Arithmetic Exception")
```

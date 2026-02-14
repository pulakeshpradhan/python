[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/10 Exception Handling/01_Introduction_to_Exception_Handling.ipynb)

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

## Exception Handling
An Exception (error) is an event due to which the normal flow of the program's instructions gets disrupted.
Errors in Python can be of the following two types i.e. Syntax errors and Exceptions.

* While exceptions are raised when some internal events occur which changes the normal flow of the program.
* On the other hand, Errors are those type of problems in a program due to which the program will stop the execution.


### Difference between Syntax Errors and Exceptions
**Syntax Error:** As the name suggests that this error is caused by the wrong syntax in the code. It leads to the termination of the program.

Example:

```python
# a = 10
# if (a > 10)
#     print("Example")
```

We will get the output as: **SyntaxError: expected ':'**. The syntax error is because there should be a ":" (colon) at the end of an if statement. Since that is not presenet in the program, it gives a syntax error.


**Exceptions:** Exceptions are raised when the program is syntactically correct but the code resulted in an error. This error does not stop the execution of the program, however, it changes the normal flow of the program.

Example:

```python
# balance = 10000
# remaining = balance / 0
# print(remaining)
```

The above example raised the **ZeroDivisionError** exception, as we are trying to divide a number by 0 which is not defined and arithmetically not possible.


### Exceptions in Python
* Python has many built-in exceptions that are raised when your program encounters an error (something in the program goes wrong).
* When these exceptions occur, the Python interpreter stops the current process and passes it to the calling process until it is handled.
* If not handled, the program will crash.
* For example, let us consider a program where we have a function A that calls function B, which in turn calls function C. If an exception occurs in function C but is not handled in C, the exception passes to B and then to A.
* If never handled, an error message is displayed and the program comes to a sudden unexpected halt.


### Some Common Exceptions
A list of common exceptions that can be thrown from a standard Python program is given below.
* ZeroDivisionError: This occurs when a number is divided by zero.
* NameError: It occurs when a name is not found. It may be local or global.
* IndentationError: It occurs when incorrect indentation is given.
* IOError: It occurs when an Input Output operation fails.
* EOFError: It occurs when the end of the file is reached, and yet operations are being performed.

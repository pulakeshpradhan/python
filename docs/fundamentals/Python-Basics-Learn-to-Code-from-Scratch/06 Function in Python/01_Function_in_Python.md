[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/06 Function in Python/01_Function_in_Python.ipynb)

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

## Function in Python

<!-- #region -->
In Python, a function is a block of reusable code that performs a specific task. Functions are used to break down large programs into smaller, more manageable pieces, and to improve code readability and reusability. A function is like a black box that can take certain input(s) as its parameters and can output a value after performing a few operations on the parameters. Functions are defined using the "def" keyword, followed by the function name and parameter list, and the code block that defines what the function does.

**Significance of functions in python:**
* Reusability: Functions can be reused in different parts of the program or in different programs altogether, which saves time and effort.

* Readability: Functions help to make code more readable by encapsulating complex logic into a single block of code with a descriptive name.

* Modularity: Functions allow us to break down a large program into smaller, more manageable parts. This makes it easier to understand and maintain the codebase.

* Debugging: Functions can be used to isolate and debug specific parts of a program, which can be helpful when troubleshooting issues.

**Basic syntax of a function:**
```python
def <function-name>(<parameters>):
    """Function's docstring"""
    <Expressions/Statements/Instructions>
```
<!-- #endregion -->

```python
## Define a function to add two numbers
def add(x, y):
    """This function returns the sum of two numbers"""
    return x + y
```

```python
add(2, 3)
```

```python
add(x=9, y=10) 
```

### Arguments and Parameters

<!-- #region -->
In Python, both parameters and arguments are used in functions to pass values between the function and the calling code. However, they have different meanings and are used in different contexts.

* Parameter: A parameter is a variable in the function definition that represents a value that must be passed to the function. It's like a placeholder that tells the function what kind of value to expect when it's called.
```python
def add(x, y):
    return x + y
```

* Argument: An argument, on the other hand, is a value that is passed to a function when it's called. It's the actual value that is used in the function when it's executed.
```python
s = add(2, 3)
```
<!-- #endregion -->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/06 Function in Python/03_Scope_of_Variables.ipynb)

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

## Scope of Variables


In a Python program, all variables may not be accessible at all location in that program. A variable's scope refers to the part(s) of the program in which the variable is accessible. A variable will only be visible to and accessible by the code blocks in its scope.

There are broadly two kinds of scope in Python:
* Global Scope: A variable declared outside of any function has global scope. This means that the variable is accessible throughout the entire program. A variable declared outside a function is known as a global variable.

* Local Scope: A variable declared inside a function is said to have local scope. This means that the variable is accessible only within the function in which it was declared. A variable declared inside a function is known as local variable.


### Creating a Global Variable

```python
x = "Global Variable"
def checkScope():
    print("x is a", x)
checkScope()
```

```python
print(x)
```

### Creating a Local Variable

```python
def checkLocal():
    y = "Local Variable"
    print("y is a",  y)
checkLocal()
```

```python
# If we call a local variable outside of its scope, we will get an error.
# print(y) 
```

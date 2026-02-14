[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/01 Introduction to Python/02_Python_Comments.ipynb)

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

## Python Comments
Comments are used in programming to explain code, make notes, and provide context for other developers who may be reading the code. In Python, comments start with the "#" symbol and continue until the end of the line. Comments are ignored by the Python interpreter, so they do not affect the execution of the code.

**Significance:**
* Comments can be used to explain Python code.
* Comments can be used to make the code more readable.
* Comments can be used to prevent execution when testing code.


### Creating a Comment
Comments start with the "#" symbol and Python interpreter will ignore them.

```python
# This is an example of how to write comment in Python.
print("Hello World")
```

### Multi-Line Comments
In Python, you can create multiline comments by enclosing them in triple quotes (''' ''') or triple double quotes (""" """). This allows you to write comments that span multiple lines, without needing to add "#" to each line

```python
"""
This is an
example of
Multi-Line Comments.
"""
print("Hello World")
```

```python
'''
This is an
example of
Multi-Line Comments too.
'''
print("Hello World")
```

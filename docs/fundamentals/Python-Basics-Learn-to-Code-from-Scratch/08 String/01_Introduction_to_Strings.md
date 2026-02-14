[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/08 String/01_Introduction_to_Strings.ipynb)

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

## String
In Python, a string is a sequence of characters enclosed within single, double or triple quotes. It is one of the most commonly used data types in Python. In this module, we will explore various operations that can be performed on strings in Python.


### String Creation
To create a string in Python, simply enclose a sequence of characters within single, double or triple quotes. For example:

```python
# Defining strings in Python

# Using single quote
name1 = 'GeoNext'
print(name1)

# using double quote
name2 = "Krishnagopal Halder"
print(name2)

# We can use triple quotes to create docstrings and/or multiline strings.
message = '''
Hello, my name is Krishnagopal.
I am modern GIS geek.
Thanks for joining today's class.
'''
print(message)

print(type(name1))
print(type(name2))
print(type(message))
```

### Accessing Characters in a String
To access individual characters in a string, we use indexing. In Python, indexing starts at 0. 

```python
print(name1[0]) # Output: G
print(name1[1]) # Output: e
print(name1[2]) # Output: o
```

### Negative Indexing
Python allows negative indexing for strings. The index of -1 refers to the last character, -2 refers to the second last character, and so on. The negative indexing starts from the last character in the string.

```python
print(name1[-1]) # Output: t
print(name1[-2]) # Output: x 
print(name1[-3]) # Output: e 
print(name1[-4]) # Output: N 
```

### String Slicing
Slicing a string means extracting a substring from the original string. We use the slice operator : for this purpose.

```python
print(name1[0:3]) # Output: Geo
print(name1[:]) # Output: GeoNext
print(name1[3:]) # Output: Next
```

```python
print(name1[-1]) # Output: t
print(name1[-4:]) # Output: Next
print(name1[-4:-1]) # Output: Nex
```

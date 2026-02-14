[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/04 Operators in Python/06_Identity_Operators.ipynb)

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

## Identity Operators


Identity operators are used to compare the memory addresses of two objects in Python. They always return a Boolean value (True or False) depending on the result of the comparison.

Here are the identity operators in Python:

* is: This operator returns True if both variables point to the same object in memory.

* is not: This operator returns True if both variables do not point to the same object in memory.


### is

```python
a = 10
b = 10
print(id(a))
print(id(b))
print(a is b)
```

```python
name1 = "GeoNext"
name2  ="GeoNext"
print(id(name1))
print(id(name2))
print(name1 is name2)
```

```python
lst1 = [1, 2, 3]
lst2 = [1, 2, 3]
print(id(lst1), id(lst2))
print(lst1 is lst2) # Output will be false since lists are mutable
```

### is not

```python
print(a is not b)
```

```python
print(name1 is not name2)
```

```python
print(lst1 is not lst2)
```

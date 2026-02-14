[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Types_of_Function.ipynb)

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

# Types of Function


## Built-in Function

```python
name = [1, 2, 3, 4, 5]
```

```python
len(name)
```

```python
print(name)
```

## Lambda Function

```python
square = lambda x: x ** 2 # One line function
```

```python
square(10)
```

```python
square(36)
```

```python
average = lambda x, y: (x + y) / 2
```

```python
average(10, 20)
```

## Function Overloading

```python
from multipledispatch import dispatch
```

```python
@dispatch(int, int)
def add(x, y):
    return x + y
```

```python
add(2, 3)
```

```python
@dispatch(int, int, int)
def add(x, y, z):
    return x + y + z
```

```python
add(2, 3)
```

```python
add(2, 3, 5)
```

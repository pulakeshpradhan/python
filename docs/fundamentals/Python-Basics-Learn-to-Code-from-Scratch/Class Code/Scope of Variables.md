[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Scope of Variables.ipynb)

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

# Scope of Variables

```python
def myName():
    name = "Local Variable"
    city = "Bankura"
    print(name)
    print(city)
```

```python
myName()
```

```python
name = "Global Variable"
```

```python
print(name)
```

```python
city = "Bankura"
```

```python
print(city)
```

## Creating a Global Variable

```python
x = "Global Variable"
def checkScope():
    print("x is a", x)
```

```python
checkScope()
```

```python
print(x)
```

## Creating a Local Variable

```python
def checkLocal():
    y = "Local Variable"
    print("y is a", y)
```

```python
checkLocal()
```

```python
# If we call a local variable outside of its scope, we will get an error.
# print(y)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Identity_Operators.ipynb)

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

# Identity Operators

```python
x = 10
y = 10
```

## is

```python
print(id(x))
print(id(y))
```

```python
print(x is y)
```

```python
name1 = "KrishnagopalHalder"
name2 = "KrishnagopalHalder"
```

```python
print(id(name1))
print(id(name2))
```

```python
print(name1 is name2)
```

## is not

```python
a = 10
b = 15
```

```python
print(id(a))
print(id(b))
```

```python
print(a is not b)
```

```python
print(name1 is not name2)
```

## Example

```python
lst1 = [1, 2, 3]
lst2 = [1, 2, 3]
```

```python
print(id(lst1), id(lst2))
```

```python
print(lst1 is lst2)
```

```python
print(lst1 is not lst2)
```

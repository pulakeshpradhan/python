[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/String_Methods.ipynb)

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

# String Methods


## Repeating/Replicating Strings

```python
name = "GeoNext"
repeat_name = name*3
print(repeat_name)
```

```python
name[::-1]
```

## slice() Constructor

```python
s1 = slice(0, 3)
s2 = slice(3, 7)
print(name[s1])
print(name[s2])
```

## String Comparison

```python
name1 = "Geo"
name2 = "Geo"
name3 = "Next"
```

```python
print(name1 == name2)
```

```python
print(name1 == name3)
```

```python
len(name1)
```

```python
len(name3)
```

```python
print(name1 < name3)
```

```python
print(name1 >= name3)
```

```python
print(name1 <= name3)
```

```python
print(name1 != name2)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Iterating on Strings.ipynb)

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

# Iterating on Strings


## Using for loop

```python
name = "Krishnagopal Halder"
```

```python
for character in name:
    print(character)
```

```python
len(name)
```

## Using for loop and range()

```python
for i in range(len(name)):# 0 - 19
    print(f"The Character {i+1}:", name[i])
    i += 1
```

## Using while loop

```python
i = 0 
while i < len(name):
    print(name[i])
    i += 1
```

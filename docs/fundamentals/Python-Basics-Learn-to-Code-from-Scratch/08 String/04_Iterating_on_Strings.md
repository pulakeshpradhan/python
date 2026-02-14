[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/08 String/04_Iterating_on_Strings.ipynb)

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

## Iterating on Strings
Iterating over strings means to access each character in the string one at a time and perform some operation on it. In Python, there are several ways to iterate over strings.


### Using for loop

```python
str1 = "Krishnagopal Halder"

# Using for loop
for i in str1:
    print(i)
```

### Using for loop and range()

```python
for i in range(len(str1)):
    print(str1[i], end=" ")
```

### Using while loop

```python
i = 0
while i < len(str1):
    print(str1[i])
    i += 1
```

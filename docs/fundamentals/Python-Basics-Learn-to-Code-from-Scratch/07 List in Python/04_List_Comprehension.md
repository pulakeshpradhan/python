[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/07 List in Python/04_List_Comprehension.ipynb)

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

## List Comprehension


List comprehension is a concise and elegant way of creating new lists based on existing lists. A list comprehension consists of an expression followed by **for** statement inside square brackets.

**Syntax**<br>
newList = ["expression" for "item" in iterable]


### Example

```python
pow2 = [2 ** x for x in range(10)]
print(pow2)
```

```python
# The above code can be written in this way as well
pow2 = []
for i in range(10):
    pow2.append(2 ** i)
print(pow2)
```

```python
# We can use other expression to modify and create a new list
newList = [x for x in range(10) if x % 2 == 1]
print(newList)
```

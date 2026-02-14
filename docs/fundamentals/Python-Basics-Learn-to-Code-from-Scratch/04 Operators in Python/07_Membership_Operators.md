[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/04 Operators in Python/07_Membership_Operators.ipynb)

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

## Membership Operators


Membership operators are used to test if a value is a member of a sequence or not. The two membership operators in Python are:

* in: This operator returns True if a value is found in the sequence, and False otherwise.

* not in: This operator returns True if a value is not found in the sequence, and False otherwise.


### in

```python
name = "GeoNext"
lst = [1, 2, 3]
print("G" in name)
print(3 in lst)
print("S" in name)
```

## not in

```python
print("S" not in name)
```

```python
print(1 not in lst)
```

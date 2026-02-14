[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/List_Comprehension.ipynb)

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

# List Comprehension

```python
myList = [1, 2, 3, 4]
```

```python
type(myList)
```

## Example

```python
pow2 = [2 ** x for x in range(10)] # range(10) 0-9
```

```python
pow2
```

```python
# The above code can be written in this way also
pow2 = []
for i in range(10):
    pow2.append(2 ** i)
print(pow2)
```

```python
# We can use other expression to modify and create a new list
oddNumbers = [x for x in range(1, 11) if x % 2 == 1]
```

```python
print(oddNumbers)
```

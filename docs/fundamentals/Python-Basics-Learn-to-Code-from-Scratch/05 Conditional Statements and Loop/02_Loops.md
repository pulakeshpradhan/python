[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/05 Conditional Statements and Loop/02_Loops.ipynb)

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

## Loops


In Python, loops are used to execute a block of code repeatedly. There are two types of loops in Python: for loops and while loops.

* For Loops: For loops are used to iterate over a sequence of values. The for loop in Python is used to iterate over a sequence (list, tuple, string, dictionary) or other iterable objects. Iterating over a sequence is called traversal.

* While Loops: While loops are used to execute a block of code as long as a condition is true. We generally use this loop when we don't know the number of times to iterate beforehand.


### for loop

```python
# Print all numbers from 0 to n
n = int(input("Enter the number yo want to print: "))

for i in range (0, n+1):
    print(i)
```

### Python range() Function


The range function is often used in for loops to generate a sequence of numbers.

```python
for i in range(5):
    print(i)
```

```python
for i in range(10, 15):
    print(i)
```

### while loop

```python
i = 0
while i < 5:
    print(i)
    i += 1
```

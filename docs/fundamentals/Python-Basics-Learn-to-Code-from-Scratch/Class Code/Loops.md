[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Loops.ipynb)

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

# Loops

```python
myList = [1, 2, 3, 4]
```

```python
for i in myList:
    print(i ** 2)
```

## for loop

```python
# Print all number from 0 to n
n = int(input("Enter the number you want to print upto: "))
```

```python
# Range function will give a sequence of number from 0 to n 
for i in range(0, 5):
    print(i)
```

```python
for i in range(0, n+1):
    print(i)
```

```python
for i in range(10, 16): # 10, 11, 12, 13, 14
    print(i)
```

## while loop

```python
i = 0 
while i < 5:
    print(i)
    i += 1
```

```python
# First iteration, i = 0
# Second iteration, i = 1
# Third iteration, i = 2
# Fourth iteration, i = 3
# Fifth iteration, i = 4
# Sixth iteration, i = 5
```

```python
num = 10

while num < 15:
    print(num, "is less than 15")
    num += 1
```

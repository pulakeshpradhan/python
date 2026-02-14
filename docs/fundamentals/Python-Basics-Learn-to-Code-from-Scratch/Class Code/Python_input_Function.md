[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Python_input_Function.ipynb)

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

# Python input() Function

```python
name = input("Write your name: ")
```

```python
print(name)
```

```python
weight = int(input("What is your weight: "))
```

```python
print(weight)
```

```python
type(weight)
```

## Taking a space-separated input in one line

```python
x, y = input("Enter the longitude and latitude: ").split()
```

```python
x = float(x)
y = float(y)
```

```python
print("The Longitude is", x)
print("The Latitude is", y)
```

```python
type(x)
```

```python
type(y)
```

# Exercise: Add two numbers

```python
num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))
```

```python
user_sum = num1 + num2
print("The sum of two numbers is", user_sum)
```

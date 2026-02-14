[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Conditional_Statement.ipynb)

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

# Conditional Statement


## if Statement

```python
# Check whether a given number is even or odd.
num = int(input("Enter a number: "))
```

```python
if (num % 2) == 0:
    print(num, "is even")
```

## if-else Statement

```python
# Check whether a given number is even or odd
num2 = int(input("Enter a number: "))
```

```python
if (num2 % 2) == 0:
    print(num2, "is even")
else:
    print(num2, "is odd")
```

## if-elif-else Statement

```python
# Find the largest among three numbers

# Taking a space-separated input from the user
x, y, z = input("Enter three numbers separated by space: ").split()
```

```python
x = int(x)
y = int(y)
z = int(z)
```

```python
if x >= y and x >= z:
    print("x is greater")
elif y >= x and y >= z:
    print("y is greater")
else:
    print("z is greater")
```

## Examples

```python
city_pop = int(input("Enter the city population: "))
```

```python
if city_pop >= 10000000:
    print("The city is a Super City.")
elif city_pop >= 5000000:
    print("The city is a Mega City.")
elif city_pop >= 1000000:
    print("The city is a Large City.")
elif city_pop >= 500000:
    print("The city is a Medium City.")
else:
    print("The city is a Small City.")
```

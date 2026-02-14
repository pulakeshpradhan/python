[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Python_print_Function.ipynb)

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

# Python print() Function

```python
print("Hello World!")
```

```python
print(100)
```

```python
# Printing more than one object
print("Kolkata",14900000,255)
```

```python
a = 15
b = 20
```

```python
print(a, b)
print(a + b)
```

```python
print("Sum of a and b is:", a+b)
```

## end Parameter

```python
print(a, end=" ")
print(b)
```

## sep Parameter 

```python
print(a, b, sep="-")
```

```python
day = 30
month = 4
year = 2023
print(day, month, year, sep="/")
```

```python
companyName = "Apple"
text = "Follow"
```

```python
# Follow@Apple
```

```python
print(text, companyName, sep="@")
```

```python
 # Follow@Apple 30/4/2023
```

```python
print("Follow@Apple 30/4/2023")
```

```python
print(text, companyName, day, month, year, sep="@")
```

## String Concatenation

```python
first_name = "Krishnagopal"
last_name = "Halder"
full_name = first_name + " " + last_name
```

```python
print(full_name)
```

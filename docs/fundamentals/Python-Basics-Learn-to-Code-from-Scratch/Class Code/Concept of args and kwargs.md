[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Concept of args and kwargs.ipynb)

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

# Concept of *args and **kwargs

```python
def add(x, y):
    return x + y 
```

```python
add(15, 10)
```

## *args (For Non-Keyword Arguments)

```python
def sum_of_numbers(*args):
    sum_of_num = 0
    for num in args:
        sum_of_num += num
    return sum_of_num
```

```python
sum_of_numbers(15, 51, 455, 17, 68, 156, 147)
```

## **kwargs (For Keyword Arguments)

```python
def myFunction(**kwargs):
    for key, value in kwargs.items():
        print(key, ":", value)
```

```python
myFunction(name="Krishnagopal Halder", city="Bankura", season="Summer")
```

```python
def sum_of_3num(x, y, z, *args):
    sum_of_num = x + y + z
    for num in args:
        sum_of_num += num
    return sum_of_num
```

```python
sum_of_3num(5, 6, 10, 15, 17, 65, 78, 10)
```

```python
def personalInfo(name, age, *args, **kwargs):
    print("The name of the person is", name)
    print("The age of the person is", age)
    if len(kwargs) > 0:
    	print("Other information")
    for key, value in kwargs.items():
        print(key, ":", value)
    if len(args) > 0:
    	print("Unknown information")
    for i in args:
        print(i)
```

```python
personalInfo("Sujoy Ghosh", 40)
```

```python
personalInfo("Ajay Sen", 28, "First arg", "Second arg", state="WB", country="India")
```

```python
personalInfo("Bijay Roy", 48, mail="abc@gmail.com", mob=6314566428, state="WB", country="India")
```

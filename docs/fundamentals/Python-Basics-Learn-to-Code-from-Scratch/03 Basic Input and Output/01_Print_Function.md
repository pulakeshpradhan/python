[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/03 Basic Input and Output/01_Print_Function.ipynb)

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

## Python print() Function


The print() function in Python is used to output text or other data to the console. It takes one or more arguments, which can be of any data type, and prints them to the console.

The basic syntax of print() is:<br>
**print(object(s), sep=separator, end=end, file=file, flush=flush)**

* object(s): One or more objects to print. This can be a string, a variable, or any otherobject.
* sep: Separator between the objects to be printed. By default, it is a space character.
* end: The character to be printed at the end of the statement. By default, it is a newline character.
* file: Output stream to which the text will be printed. By default, it is the console.
* flush: A Boolean value that specifies whether to flush the output buffer. By default, it is False.

```python
print("Hello World")
```

```python
print("Hello, GIS lovers!")
```

```python
print("GeoNext is best for GIS content")
```

### Printing more than one object

```python
print("Hello", "How are you?")
```

```python
a = 5
b = 10
print(a, b)
```

### Python end parameter in print()

```python
print("Welcome to", end=" ")
print("GeoNext")
```

```python
print("Follow ", end="@")
print("GeoNext")
```

### Python sep parameter in print()

```python
print("09", "03", "2023", sep="-")
```

```python
print("krish", "GeoNext", sep="@")
```

### String Concatenation

```python
print("a"+"bc")
```

```python
print("Geo" + "Next")
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/08 String/02_String_Concatenation.ipynb)

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

## String Concatenation
String concatenation is the process of combining two or more strings into a single string. String concatenation can be done using many ways.
We can perform string concatenation using the following ways:
* Using + operator 
* Using join() method 
* Using % operator 
* Using format() function


### Using + Operator
The simplest way to concatenate two strings in Python is to use the + operator. The + operator functions is similar to the arithmetic + operator, but here both the operands must be string.

```python
first_name = "Krishnagopal"
last_name = "Halder"
print(first_name + " " + last_name)
```

### Using join() Method
The join() method is a powerful and efficient way to concatenate strings in Python. It takes an iterable object (such as a list or tuple) and joins the elements of the iterable with the string. If the iterable contains any non-string values, it raises a TypeError exception.

```python
words = ["Krishnagopal", "Halder", "is", "a", "student", "of", "Geoinformatics"]
print(" ".join(words))
name = " ".join([first_name, last_name])
print(name)
```

### Using % Operator
We have used % operator for string formatting, but it can also be used for string concatenation. % operator helps both in string concatenation and string formatting.

```python
n1 = "Geo"
n2 = "Next"
print("%s%s"%(n1, n2))
```

### Using format() function
The format() function is a powerful way to format strings in Python. It provides a flexible and efficient way to combine strings and variables into a single string.

```python
name = "Krishnagopal Halder"
course = "Python"
message = "Hello, {}. Thank you for joining this {} online course.".format(name, course)
print(message)
```

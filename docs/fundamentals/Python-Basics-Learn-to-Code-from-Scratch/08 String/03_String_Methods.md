[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/08 String/03_String_Methods.ipynb)

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

## String Methods


### Repeating/Replicating Strings
In Python, we can repeat or replicate strings using the multiplication operator *. The multiplication operator takes two operands: a string and an integer. When we multiply a string by an integer, Python returns a new string that contains the original string repeated the specified number of times.

```python
name = "GeoNext"
print(name * 3)
```

### String Slicing
String slicing is the process of extracting a part of a string. In Python, we can slice a string using two methods.
* Extending Indexing
* slice() Constructor


#### Extending Indexing
In Python, we can slice a string using the indexing operator []. The indexing operator takes one or two arguments: the index of the character we want to extract, or the start and end indices of the slice we want to extract.

```python
name = "GeoNext"
print(name[3])
print(name[0:3])
print(name[3:])
print(name[::2])
```

```python
# Print string in reverse
print(name[::-1])
```

#### slice() Constructor
Python provides a built-in function called slice() that we can use to create slice objects. Slice objects can then be passed as arguments to the indexing operator to slice a string.<br>
**Syntax:**
slice(start, stop, step)
* start: Optional. An integer that specifies the starting index of the slice. If omitted, it defaults to None, which means the slice starts at the beginning of the string.
* stop: Required. An integer that specifies the ending index of the slice. This index is not included in the slice.
* step: Optional. An integer that specifies the step size of the slice. If omitted, it defaults to None, which means the slice uses a step size of 1.

```python
name = "GeoNext"
s1 = slice(0, 3)
s2 = slice(3, 8)
s3 = slice(0, 8, 2)
print(name[s1])
print(name[s2])
print(name[s3])
```

### String Comparison
Comparing strings is the process of comparing two or more strings to determine whether they are equal or not. String comparison in Python takes place character by character i.e. each character in the same index, are compared with each other.

There are several ways to compare strings in Python. One way is to use the comparison operators such as ==, !=, <, <=, >, and >=. These operators compare the strings lexicographically and return a Boolean value of True or False depending on the comparison result.

```python
str1 = "Krishnagopal"
str2 = "Krishnagopal"
str3 = "Halder"
print(str1 == str2) # Checks if two strings are equal or not
print (str1 != str3) # Checks if two strings are not equal
print(str1 < str3) # Checks if the first string is less than the second string
print(str1 <= str3) # Checks if the first string is less than or equal to the second string
print(str1 > str3) # Checks if the first string is greater than the second string
print(str1 >= str3) # Checks if the first string is greater than or equal to the second string
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/02 Variables and Datatypes/02_Introduction_to_Datatypes.ipynb)

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

## Introduction to Datatypes
In Python, data types are used to categorize and represent different types of data that can be used in a program. They are used to classify different types of data that can be stored and manipulated in a program. The most common data types in Python include integers, floats, strings, booleans, and lists.

The most common built-in data types in Python are:
* **Integer:** Represents whole numbers, such as 1, 2, 3, -4, -5, etc.

* **Float:** Represents decimal numbers, such as 3.14, 2.5, -1.0, etc.

* **String:** Represents a sequence of characters, such as "hello", "world", "123", etc.

* **Boolean:** Represents either True or False.

* **List:** Represents an ordered collection of values, which can be of any data type. For example, [1, 2, 3], ["apple", "banana", "orange"], etc.

* **Tuple:** Similar to a list, but immutable, meaning it cannot be modified once created. For example, (1, 2, 3), ("apple", "banana", "orange"), etc.

* **Set:** Represents an unordered collection of unique values. For example, {1, 2, 3}, {"apple", "banana", "orange"}, etc.

* **Dictionary:** Represents a collection of key-value pairs, where each key is associated with a value. For example, {"name": "John", "age": 30}, {"fruit": "apple", "color": "red"}, etc


### Examples


### Integer Datatype
Integers are whole numbers, such as 1, 2, 3, etc. They can be positive or negative.

```python
# Integer Datatype
myInt = 8
print(myInt, "is a type of", type(myInt))
```

### Float Datatype
Floats are numbers with a decimal point, such as 3.14, 2.5, etc. They can also be positive or negative.

```python
# Float Datatype
myFloat = 8.5
print(myFloat, "is a type of", type(myFloat))
```

### String Datatype
Strings are sequences of characters, such as "hello", "world", etc. They are enclosed in quotation marks (single or double).

```python
# String Datatype
myString1 = "GeoNext"
myString2 = "8.5"
print(myString1, "is a type of", type(myString1))
print(myString2, "is a type of", type(myString2))
```

### Boolean Datatype
Booleans are values that can be either True or False. They are often used in conditional statements and loops.

```python
# Boolean Datatype
myBool1 = True
myBool2 = False
print(myBool1, "is a type of", type(myBool1))
print(myBool2, "is a type of", type(myBool2))
```

### List Datatype
Lists are ordered collections of values, which can be of any data type. They are enclosed in square brackets and separated by commas, like [1, 2, 3] or ["apple", "banana", "orange"].

```python
# List Datatype
myList1 = ["Apple", "Banana", "Orange"]
myList2 = [1, 2, 3] 
myList3 = ["Apple", 2, 3.5]  
print(myList1, "is a type of", type(myList1))
print(myList2, "is a type of", type(myList2))
print(myList3, "is a type of", type(myList3))
```

### Tuple Datatype
Tuples are similar to a list, but immutable, meaning it cannot be modified once created. They are enclosed in round brackets and separated by commas, such as (1, 2, 3) or ("apple", "banana", "orange").

```python
# Tuple Datatype
myTuple1 = ("Apple", "Banana", "Orange")
myTuple2 = (1, 2, 3)
myTuple3 = ("Apple", 2, 3.5)
print(myTuple1, "is a type of", type(myTuple1))
print(myTuple2, "is a type of", type(myTuple2))
print(myTuple3, "is a type of", type(myTuple3))
```

### Set Datatype
Set represents an unordered collection of unique values. They are enclosed in curly brackets and separated by commas. For example, {1, 2, 3}, {"apple", "banana", "orange"}, etc.

```python
# Tuple Datatype
mySet1 = {"Apple", "Banana", "Orange"}
mySet2 = {1, 2, 3, 2}
mySet3 = {"Apple", 2, 3.5, "Apple"}
print(mySet1, "is a type of", type(mySet1))
print(mySet2, "is a type of", type(mySet2))
print(mySet3, "is a type of", type(mySet3))
```

### Dictionary Datatype
It represents a collection of key-value pairs, where each key is associated with a value. Dictionaries are enclosed with curly brackets and the items of the dictionary are separated by commas. For example, {"name": "John", "age": 30}, {"fruit": "apple", "color": "red"}, etc.

```python
# Dictionary Datatype
myDict1 = {"fruit": "Apple", "color": "Red", "is_edible": True}
print(myDict1, "is a type of", type(myDict1))
```

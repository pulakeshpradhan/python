[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/02 Variables and Datatypes/01_Introduction_to_Variables.ipynb)

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

## Introduction to Variables
Variables in Python are containers that store data values. They are used to hold and manipulate data in a program. Python uses a dynamic typing system, which means that you don't need to declare the data type of a variable before using it.

Variables are fundamental to programming for two reasons: 
* **Variables keep values accessible:** For example,The result of a time-consuming operation can be assigned to a variable so that the operation need not be performed each time we need the result.

* **Variables give values context:** For example, The number 56 could mean lots of different things, such as the number of students in a class, or the average weight of all students in the class. Assigning the number 56 to a variable with a name like num_students would make more sense, to distinguish it from another variable average_weight, which would refer to the average weight of the students. This way we can have different variables pointing to different values.


### How are Values Assigned to A Variable?
In Python, you can assign values to variables using the assignment operator (=). The assignment operator assigns the value on the right-hand side of the operator to the variable on the left-hand side of the operator.

For Example:Now let us create a variable namely myNum to hold a specific number.

```python
# Creating a new variable
myNum = 4
myNum
```

<center><img src="https://miro.medium.com/v2/resize:fit:1400/1*QT3V91g69EUI7Uv0GY1nsg.png" style="max-width: 450px; height: auto"></center>

```python
# Creating multiple variables
myInt = 4
myReal = 2.5
myChar = "a"
myString = "hello"
print(myInt)
print(myReal)
print(myChar)
print(myString)
```

<center><img src="https://miro.medium.com/v2/resize:fit:1400/1*Px7h03Ih7B5QZu4KQpSEoQ.png" style="max-width: 850px; height: auto"></center> 


### Naming a Variable
When naming a variable in Python, there are some rules and conventions you should follow to make your code more readable and understandable. Here are some guidelines for naming variables:

* Variable names must start with a letter or underscore (_), but cannot start with a number.
* Variable names can only contain letters, numbers, and underscores (_). They cannot contain spaces or special characters.
* Variable names should be descriptive and meaningful, so that other programmers can understand what the variable represents.
* Variable names are case-sensitive. For example:- The variable names Temp and temp are different
* Variable names should be written in lowercase, with words separated by underscores (_). This convention is called "snake_case" and is widely used in Python.
* Avoid using reserved keywords as variable names, such as "if", "while", "for", "and", "or", etc.


### Examples

```python
# Correct Variables
a1 = 5
_b2 = 10
b = 10
```

```python
# Incorrect Variables
# 1a = 5
# 23b = 10
# 1@ = 5
```

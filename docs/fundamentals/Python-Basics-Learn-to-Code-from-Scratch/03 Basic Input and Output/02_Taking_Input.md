[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/03 Basic Input and Output/02_Taking_Input.ipynb)

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

## Python input() Function


The input() function in Python is used to accept user input from the keyboard. It prompts the user to enter a value, reads the input from the user, and returns it as a string.

The basic syntax of input() is: <br>
**input(prompt)**

* prompt: A string that is displayed to the user as a prompt for input. It is an optional argument, and if it is not provided, the function will display an empty prompt.

```python
name = input("Enter your name: ")
print(name)
```

```python
print(type(name))
```

When we use the input() function in Python to get user input, the input is always returned as a string. If we want to perform any numerical operations on the input, we will need to convert it to an integer using the int() function.

```python
num = input("Enter a number: ")
print(num, type(num))
```

```python
num = int(input("Enter a numbert: "))
print(num, type(num))
```

### Taking a space-separated input in one line

```python
x, y = input("Enter the longitude and latitude: ").split()
print(x)
print(y)
```

### Exercise: Add two numbers 

```python
num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))
s = num1 + num2
print("The sum of two numbers is", s)
```

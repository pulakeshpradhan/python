[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/05 Conditional Statements and Loop/01_Conditional_Statement.ipynb)

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

## Conditional Statement


In Python, conditional statements are used to execute certain code if a particular condition is true. The most commonly used conditional statements in Python are if, elif, and else. This course material will cover the syntax and usage of these statements.

* If Statement: The if statement is used to execute code if a specific condition is true.

* If-Else Statement: The if-else statement is used to execute one block of code if a condition is true and another block of code if it is false. 

* If-Elif-Else Statement: The if-elif-else statement is used to execute one block of code if the first condition is true, another block of code if the second condition is true, and so on. If none of the conditions are true, then the code inside the else block will be executed. 


### If Statement

```python
# Check whether a given number is even or odd.
num = int(input("Enter a number: "))

if num % 2 == 0:
    print(num, "is even.")
```

### If-Else Statement

```python
# Check whether a given number is even or odd.
num = int(input("Enter a number: "))

if num % 2 == 0:
    print(num, "is even.")
else:
    print(num, "is odd.")
```

### If-Elif-Else Statement

```python
# Find the largest among three numbers
a = 10
b = 8
c = 15

if a >= b and a >= c:
    print("a is greater.")
elif b >= a and b >= c:
    print("b is greater.")
else:
    print("c is greater.")
```

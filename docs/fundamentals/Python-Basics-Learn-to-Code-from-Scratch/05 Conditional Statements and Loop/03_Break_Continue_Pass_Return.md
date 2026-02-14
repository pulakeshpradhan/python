[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/05 Conditional Statements and Loop/03_Break_Continue_Pass_Return.ipynb)

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

## Break, Continue, Pass and Return Statement


Introduction In Python programming, there are several types of control flow statements that allow you to alter the normal flow of program execution. These include the break, continue, pass, and return statements. In this course material, we will discuss each of these statements and how they can be used in Python programming.

* The break Statement: The break statement is used to prematurely exit out of a loop. It works with the for loop and the while loop. When a break statement is executed within a loop, the loop is immediately terminated, and the program continues to run from the first statement after the loop. The break statement is typically used to stop a loop when a certain condition has been met.

* The continue Statement: The continue statement is used to skip to the next iteration of a loop without executing any statements within the loop for the current iteration. It works with the for loop and the while loop. The continue statement is typically used to skip over certain values in a loop.

* Pass Statement: The pass statement is a null operation statement, meaning that it does nothing. It is used as a placeholder when there is no code to be executed in a certain part of a program.

* Return Statement: The return statement is used to end the execution of a function and return a value to the caller. When the interpreter reaches the return statement, it immediately exits the function and returns control back to the caller with the value provided.


### Break Statement

```python
i = 0
print("Program started.")

while i < 5:
    if i == 3:
        print("Executing break statement in the next statement.")
        break
    print(i)
    i += 1
    
print("Program ended.")
```

### Continue Statement

```python
i = 0
print("Program started.")

while i < 5:
    if i == 3:
        i += 1
        continue
    print(i)
    i += 1
    
print("Program ended.")
```

### Pass Statement

```python
i = 0
print("Program started.")

while i < 5:
    if i == 3:
        i += 1
        pass
    else:
        print(i)
        i += 1
        
print("Program ended.")
```

### Return Statement

```python
# Factorial of nth number
def factorial(n):
    fac = 1
    if n == 0 or n == 1:
        return 1
    else:
        for i in range(1, n+1):
            fac = fac * i
        return fac
```

```python
factorial(5)
```

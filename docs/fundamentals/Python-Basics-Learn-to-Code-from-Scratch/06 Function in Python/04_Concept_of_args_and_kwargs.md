[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/06 Function in Python/04_Concept_of_args_and_kwargs.ipynb)

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

## Concept of *args and **kwargs


In Python, it's possible to define functions that accept a variable number of arguments. This can be useful when we don't know in advance how many arguments a function will need to handle, or when we want to make our code more flexible. Python uses these special symbols for passing arguments:

* *args (For Non-Keyword Arguments)
* **kwargs (For Keyword Arguments)


### *args (For Non-Keyword Arguments)


In Python, *args is a special syntax used to pass a variable number of positional arguments to a function. The *args notation allows us to pass any number of arguments to a function, and those arguments will be collected into a tuple.

For example, if we want to make an addition function that supports taking any number of arguments and able to add them all together. In this case we can use *args.

One important thing to note is that *args must come at the end of a function's parameter list, after any named arguments. This is because *args collects all positional arguments that are not matched to named parameters, so any named parameters that come after *args will be ignored

```python
def sum(*args):
    s = 0
    for i in args:
        s += i
    print(s)
```

```python
sum(1, 2, 3, 4, 5)
```

### **kwargs (For Keyword Arguments)


In Python, **kwargs is a special syntax used to pass a variable number of keyword arguments to a function. The **kwargs notation allows us to pass any number of keyword arguments to a function, and those arguments will be collected into a dictionary.

A keyword argument is where we provide a name to the variable as we pass it into the function.

One important thing to note is that **kwargs must come after *args in a function's parameter list, if both are used. This is because *args collects all positional arguments that are not matched to named parameters, while **kwargs collects all keyword arguments that are not matched to named parameters.

```python
def myFunction(**kwargs):
    for key, value in kwargs.items():
        print(key, ":", value)
```

```python
myFunction(name="Krishnagopal Halder", age=22)
```

### Example with *args and **kwargs

```python
def personalInfo(name, age, *args, **kwargs):
    print("The name of the peron is", name)
    print("The age of the person is", age)
    print("Other informations:")
    for key, value in kwargs.items():
        print(key, ":", value)
    print("Unknown informations:")
    for i in args:
        print(i)    
```

```python
personalInfo("Krishnagopal Halder", 22, "First arg", "Second arg", state="WB", country="India")
```

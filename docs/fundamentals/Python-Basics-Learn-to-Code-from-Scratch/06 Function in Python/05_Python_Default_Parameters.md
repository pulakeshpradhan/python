[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/06 Function in Python/05_Python_Default_Parameters.ipynb)

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

## Python Default Parameters


In Python, function parameters can have default values. We can provide a default value to a parameter by using the assignment(=) operator.

```python
def programmer(name, age, coreLang="Python"):
    print("The name of the programmer is", name)
    print("The age of the programmer is", age)
    print("The core programming language is", coreLang)
```

```python
# Calling function with the default value of coreLang parameter
programmer("Krishnagopal Halder", 22)
```

```python
# Calling function with the changed value of coreLang parameter
programmer("Arav Mahanti", 25, "C++")
```

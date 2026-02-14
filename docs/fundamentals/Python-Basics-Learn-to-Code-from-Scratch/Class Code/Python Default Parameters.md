[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Python Default Parameters.ipynb)

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

# Python Default Parameter

```python
def programmer(name, age, coreLang="Python"):
    print("The name of the programmer is", name)
    print("The age of the programmer is", age)
    print("The core programming language is", coreLang)
```

```python
# Calling the funtion with the deafult value of coreLang parameter
programmer("Abir Sen", 26)
```

```python
# Calling function with the changed value of coreLang parameter
programmer("Abir Sen", 26, "C++")
```

```python
programmer(name="Abir Sen", age=26, coreLang="C++")
```

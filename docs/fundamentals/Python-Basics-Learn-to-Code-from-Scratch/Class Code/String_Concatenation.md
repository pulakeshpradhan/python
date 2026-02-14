[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/String_Concatenation.ipynb)

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

# String Concatenation


## Using + Operator

```python
name1 = "Geo"
name2 = "Next"
```

```python
full_name = name1 + name2
full_name
```

```python
first_name = "Krishnagopal"
last_name = "Halder"
```

```python
full_name = first_name + " " + last_name
full_name
```

## Using join() Method

```python
words = ["Here", "I", "am", "concatinating", "multiple", "strings", "with", "join", "method."]
message = " ".join(words)
print(message)
print(type(words))
print(type(message))
```

## Using % Operator

```python
name1 = "Geo"
name2 = "Next"
print("%s%s"%(name1, name2))
```

## Using format() function

```python
name = "Krishnagopal"
course_name = "Python"
message = "Hello {}, Thank you for joining today's {} class.".format(name, course_name)
print(message)
```

```python
# Lets use format function with user inputs.
name = input("What is your name?")
course_name = input("In which course you want to enroll?")
message = "Hello {}, Thank you for showing interest in {} online course.".format(name, course_name)
print(message)
```

## f Strings

```python
message = f"Hello {name}, Thank you for showing interest in {course_name} online course."
print(message)
```

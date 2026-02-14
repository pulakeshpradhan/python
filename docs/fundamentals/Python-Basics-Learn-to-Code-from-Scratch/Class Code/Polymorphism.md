[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Polymorphism.ipynb)

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

## Polymorphism


### Example 1: Polymorphism in addition(+) operator

```python
a = 5 # int datatype
b = 10
```

```python
a + b
```

```python
type(a)
```

```python
str1 = "Geo"
str2 = "Next"
```

```python
type(str1)
```

```python
str1 + str2
```

### Example 2: Functional Polymorphism in Python

```python
list1 = [1, 2, 3, 4, 5]
str1 = "GeoNext"
dict1 = {"a": 10, "b": 20}
```

```python
print(len(list1))
print(len(str1))
print(len(dict1))
```

## Class Polymorphism in Python

```python
class Male:
    
    # constructor
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def info(self):
        print("Hello, I am Male.")
        print(f"My name is {self.name}.")
        print(f"I am {self.age} years old.")
```

```python
class Female:
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def info(self):
        print("Hello, I am Female.")
        print(f"My name is {self.name}.")
        print(f"I am {self.age} years old.")
```

```python
# Creating instances
male1 = Male("Ayan Pal", 48)
female1 = Female("Sayani Pal", 40)
```

```python
# Running a loop over the set of objects
# Calling the info() function common to both
for human in (male1, female1):
    human.info()
    print("\n")
```

### Polymorphism and Inheritance

```python
# Create a parent / super class
class Human:
    def __init__(self, name):
        self.name = name
        
    def info(self):
        print("I am a Human.")
        print(f"My name is {self.name}")
```

```python
# Create two child class
class Male(Human):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age
    
    def info(self):
        print("Hello, I am a Male.")
        print(f"My name is {self.name}.")
        print(f"I am {self.age} years old.")
```

```python
class Female(Human):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age
        
    def info(self):
        print("Hello, I am a Female.")
        print(f"My name is {self.name}.")
        print(f"I am {self.age} years old.")
```

```python
# Creating instances
human1 = Human("Ayan Roy")
male1 = Male("Bijay Pal", 28)
female1 = Female("Jayita Sen", 25)
```

```python
for i in (human1, male1, female1):
    i.info()
    print("\n")
```

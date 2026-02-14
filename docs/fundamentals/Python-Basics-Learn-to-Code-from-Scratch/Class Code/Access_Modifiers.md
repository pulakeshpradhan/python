[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Access_Modifiers.ipynb)

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

# Access Modifiers


## Public Modifier

```python
# Creating a student class with public member
# member = method + variables inside of class
class Pub_Student:
    name = None # public member by default
    age = None # public member
    
    # constructor
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def public_method(self): # public member
        print("This is a public method.")
```

```python
# Creating objects/instances from the Pub_Student class
pub_obj = Pub_Student("Sovon Roy", 22)
```

```python
# Calling the public members of the class
print(pub_obj.name)
print(pub_obj.age)
pub_obj.public_method()
```

## Protected Modifier

```python
# Creating a student class with protected member
class Pro_Student:
    name = None # public member
    _age = None # protected member
    
    # constructor
    def __init__(self, name, age):
        self.name = name
        self._age = age
        
    def print_details(self): # publc member (method)
        print(f"Student Name: {self.name}")
        print(f"Student age: {self._age}")
        
    def _protected_method(self): # protected member (method)
        print("This is a protected method.")
```

```python
# Creating objects/instances from the Pro_Student class
pro_obj = Pro_Student("Anil De", 14)
```

```python
# Calling the protected members of the class
print(pro_obj._age)
pro_obj._protected_method()
```

```python
# Calling the public members of the class
print(pro_obj.name)
pro_obj.print_details()
```

## Private Modifier

```python
# Creating a student class with private member
class Pri_Student:
    name = None # public member
    __age = None # private member
    
    # constructor
    def __init__(self, name, age):
        self.name = name
        self.__age = age
        
    def print_details(self): #public member
        print(f"Student Name: {self.name}")
        print(f"Student Age: {self.__age}")
        
    def __private_method(self): # private member
        print("This is a private method")
```

```python
# Creating objects/instances from the Pri_Student class
pri_obj = Pri_Student("Bikram Roy", 16)
```

```python
# Calling the public members of the class
print(pri_obj.name)
pri_obj.print_details()
```

```python
# Calling the private members of the class
# pri_obj.__age
# pri_obj.__private_method()
```

### Name Mangling

```python
pri_obj._Pri_Student__age
```

```python
pri_obj._Pri_Student__private_method()
```

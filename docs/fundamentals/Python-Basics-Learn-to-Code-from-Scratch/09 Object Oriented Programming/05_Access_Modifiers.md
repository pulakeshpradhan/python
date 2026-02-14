[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/09 Object Oriented Programming/05_Access_Modifiers.ipynb)

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

## Access Modifiers
Access Modifiers are the specifications that can be defined to fix boundaries for classes when accessing member functions (methods) or member variables are considered. Various object-oriented languages like C++, Java, Python control access modifications which are used to restrict access to the variables and methods of the class. Most programming languages majorly have the following three forms of access modifiers, which are **Public**, **Private**, and **Protected** in a class.


<center><img src="https://miro.medium.com/v2/resize:fit:1162/1*AsKRlD4xL50sqSDAOYvhMA.jpeg" style="max-width: 550px; height: auto"></center>


### Public Modifier
The members of a class that are declared public are easily accessible from any part of the program. By default, all members (attributes and methods) in a Python class are considered public and can be accessed from anywhere. Consider the given example:

```python
# Creating a student class with public member
class Pub_Student:
    name = None # public member by default
    age = None # public member
    
    # constructor
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def public_method(self):
        print("This is a public method.")
```

```python
# Creating objets/instances from Student class
pub_obj = Pub_Student("Sohon Roy", 16)
```

```python
# Calling the public members of the class
print(pub_obj.name)
print(pub_obj.age)
pub_obj.public_method()
```

### Protected Modifier
The members (attributes and methods) of a class that are declared protected are only accessible to a class derived from it. Data members of a class are declared **protected** by adding a single underscore '_' symbol before the data member of that class. However, this naming convention does not enforce actual access restrictions. It serves more as a hint to other developers that a member should be treated as protected. Here's an example:

```python
# Creating a student class with protected member
class Pro_Student:
    _name = None # protected data member
    age = None # public data member
    
    # constructor
    def __init__(self, name, age):
        self._name = name
        self.age = age
        
    def _protected_method(self):
        print("This is a protected method.")
```

```python
# Creating objets/instances from Student class
pro_obj = Pro_Student("Ayan Ghosh", 16)
```

```python
# Calling the protected members of the class
print(pro_obj._name)
pro_obj._protected_method()
```

```python
# Calling the public members of the class
print(pro_obj.age)
```

### Private Modifier
The member of a class that are declared **private** are accessible within the class only. A private access modifier is the most secure access modifier. Data members of a class are declared private by adding a double underscore '__' symbol before the data member of that class. Here's an example: 

```python
# Creating a student class with private member
class Pri_Student:
    name = None # public data member
    __age = None # private data member
    
    # constructor
    def __init__(self, name, age):
        self.name = name
        self.__age = age
        
    def __private_method(self):
        print("This is a private method.")
```

```python
# Creating objets/instances from Student class
pri_obj = Pri_Student("Bikash Dey", 21)
```

```python
# Calling the public members of the class
print(pri_obj.name)
```

```python
# Calling the private members of the class
#print(pri_obj.__age)
```

We will get an **AttributeError** when we try to access the **'__age'** attribute. This is because the **__age** is a private attribute and hence it cannot be accessed from outside the class.


**Name Mangling:**<br>
Private members in Python are name-mangled, which means their names are modified to make them harder to access. The name-mangling scheme is as follows: <br>a double underscore __ at the beginning of an attribute or method name is replaced with _ClassName, where ClassName is the name of the class.

```python
# Accessing the private members of the class
print(pri_obj._Pri_Student__age)
pri_obj._Pri_Student__private_method()
```

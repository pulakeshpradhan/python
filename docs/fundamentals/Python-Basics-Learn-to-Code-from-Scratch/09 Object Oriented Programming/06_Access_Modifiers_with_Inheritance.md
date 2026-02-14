[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/09 Object Oriented Programming/06_Access_Modifiers_with_Inheritance.ipynb)

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

## Access Modifiers with Inheritance
Inheritance in object-oriented programming allows a subclass (derived class) to inherit the attributes and methods of a superclass (base class). When it comes to access modifiers, the behavior is slightly different depending on the specific programming language. Let's discuss how access modifiers work with inheritance in Python:


1. **Public Members:** Public members of the superclass are accessible by the subclass. They retain their public visibility in the subclass.

2. **Protected Members:** Protected members of the superclass can be accessed by the subclass, similar to public members. However, it is still a convention and not enforced by the language. Even though the protected members are intended to be protected, it can still be accessed and invoked by the Subclass. However, it is important to respect the convention and consider it as a hint to other developers.

3. **Private Members:** Private members of the superclass are not directly accessible by the subclass. However, it is possible to access them indirectly by defining public or protected methods in the superclass that provide access to the private members.

```python
# Creating a super/parent class
class Vehicle:
    def __init__(self, brand, name, color, product_no):
        self.brand = brand # public member
        self.name = name # public member
        self.color = color # public member
        self.__product_no = product_no # private member
        
    def _set_owner(self, owner, phone): # protected member
        self._owner = owner
        self._phone = phone
        
    def _print_owner_details(self): # protected member
        print(f"Owner Name: {self._owner}")
        print(f"Phone No: {self._phone}")
        
    def print_vehicle_details(self): # public member
        print(f"Vehicle Brand: {self.brand}")
        print(f"Vehicle Name: {self.name}")
        print(f"Color: {self.color}")
        
    def __show_product_no(self): # private method
        print(f"Vehicle Product No: {self.__product_no}")
```

```python
# Creating a sub/child class
class Car(Vehicle):
    def __init__(self, brand, name, color):
        super().__init__(brand, name, color, None)
```

```python
# Creating instances from the child class
car1 = Car("Maruti", "Maruti 800", "White")
```

```python
# Calling the public members
car1.print_vehicle_details()
```

```python
# Calling the protected members
car1._set_owner("Subrata Dey", 6245614786)
car1._print_owner_details()
```

```python
# Calling the private members
# It will throw an error
# car1.__product_no 
# car1._Car__product_no
# car1.__show_product_no()
```

```python
# Creating an instance of the super/parent class
vehicle1 = Vehicle("Ferrari", "Ferrari XZ", "Black", 1001)
```

```python
# Calling the private members
vehicle1._Vehicle__show_product_no()
```

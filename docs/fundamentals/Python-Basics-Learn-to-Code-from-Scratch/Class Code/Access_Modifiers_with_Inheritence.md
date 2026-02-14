[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Access_Modifiers_with_Inheritence.ipynb)

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
        
    def print_vehicle_details(self): #public member
        print(f"Vehicle Brand: {self.brand}")
        print(f"Vehicle Name: {self.name}")
        print(f"Vehicle Color: {self.color}")
        
    def __show_product_no(self): # private member
        print(f"Vehicle Product No: {self.__product_no}")
```

```python
#  Create an instance from the Vehicle class
vehicle1 = Vehicle("Maruti", "Maruti 800", "White", 1001)
```

```python
vehicle1.print_vehicle_details()
```

```python
vehicle1._set_owner("Ajay Ghosh", 6254864958)
```

```python
vehicle1._print_owner_details()
```

```python
vehicle1._Vehicle__product_no
```

```python
vehicle1._Vehicle__show_product_no() # Name Mangling
```

```python
# Creating a sub/child class
class Car(Vehicle):
    def __init__(self, brand, name, color):
        super().__init__(brand, name, color, None)      
```

```python
# Create an instance from Car class
car1 = Car("Ferrari", "Ferrai XZ", "Black")
```

```python
car1.print_vehicle_details()
```

```python
car1._set_owner("Dip Ghosh", 7182648975)
```

```python
car1._print_owner_details()
```

```python
# car1._Car__product_no 
# This will throw an error. Private members cannot be accessed from outside the class.
```

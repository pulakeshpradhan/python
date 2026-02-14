[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/09 Object Oriented Programming/02_Constructor.ipynb)

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

<!-- #region -->
## Constructor
Constructors are special methods in Python classes that are used to initialize objects. The constructor method is automatically called when an object is created from a class. In Python, the constructor method is defined using the __init__() method.

**Syntax of a constructor:**
```python
def __init__(self):
    # body of the constructor
```
<!-- #endregion -->

### Types of constructors
* **Default Constructor:**
A default constructor is provided by Python if no constructor is explicitly defined in a class. It takes no parameters and doesn't perform any initialization. Its definition has only one argument which is a reference to the instance being constructed known as self. <br>
Example: def __init__(self):

* **Parameterized Constructor:**
A parameterized constructor is defined with one or more parameters to initialize the object's attributes during creation. It allows you to pass values at the time of object creation to set initial attribute values.<br>
Example: def __init__(self, parameter1, parameter2, ...):


### The self Parameter
The self parameter is a convention used in method definitions within a class. It represents the instance of the object on which the method is being called. It acts as a reference to the current object and allows access to its attributes and methods.

* The self parameter is a reference to the current instance of the class and is used to access variables that belong to the class.
* It does not have to be named 'self', you can call it whatever you like, but it has to be the first parameter of any function in the class.
* You can give .__init__() any number of parameters, but the first parameter will always be a variable called self.


### Defining a Class with Constructor

```python
# Creating a car class with parameterized constructor
class Car:
    def __init__(self, name, topspeed, color):
        self.name = name
        self.topspeed = topspeed
        self.color = color
```

```python
# Creating an instance (object) from that car class
car1 = Car("Ferrari", 400, "Red")
```

```python
# Checking the attribute of the car object
print("The name of the car is", car1.name)
print("The topspeed of the car is", car1.topspeed, "Km/h")
print("The color of the car is", car1.color)
```

```python
# Add a function to the car class that will print the car attributes
class Car:
    def __init__(self, name, topspeed, color):
        self.name = name
        self.topspeed = topspeed
        self.color = color
    def car_info(self):
        print("The name of the car is", self.name)
        print("The topspeed of the car is", self.topspeed, "Km/h")
        print("The color of the car is", self.color)
```

```python
# Creating an object of the car class
car2 = Car("Maruti 800", 120, "White")
# Calling the car_info function to the object
car2.car_info()
```

### Example:
An example of a class for an application form in Python:

```python
class  ApplicationForm:
    def __init__(self, name, age, dob, phone):
        self.name = name
        self.age = age
        self.dob = dob
        self.phone = phone
        
    def set_email(self, email):
        self.email = email
        
    def set_address(self, address):
        self.address = address
        
    def print_form(self):
        print("Application Form:")
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"DOB: {self.dob}")
        print(f"Phone: {self.phone}")
        print(f"Email: {self.email}")
        print(f"Address: {self.address}")
```

```python
form1 = ApplicationForm("Kunal Roy", 20, "29-08-2002", 6245684125)
form1.set_email("roy_kunal2000@gmail.com")
form1.set_address("Newtown, Kolkata, West Bengal, India")
```

```python
form1.print_form()
```

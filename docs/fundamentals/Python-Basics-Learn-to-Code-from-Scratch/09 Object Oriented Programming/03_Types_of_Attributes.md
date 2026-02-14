[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/09 Object Oriented Programming/03_Types_of_Attributes.ipynb)

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

## Types of Attributes
In Python, attributes are variables that belong to an object or class. They store data that defines the characteristics or properties of the object. There are two types of attributes commonly used in Python:


### Instance Attributes:
Attributes created in .__init__() are called instance attributes. An instance attribute's value is specific to a particular instance of the class. All car objects have a name and a topspeed, but the values for the name and topspeed will vary depending on the car instance. Different objects of the Car class have different names and top speeds.<br>

**Characteristics:**
* Instance attributes are specific to each instance of a class.
* They are defined and assigned within the constructor (__init__() method) using the self keyword.
* Each object of the class can have different values for these attributes.
* Example: self.name, self.topseed, etc.


```python
# Example of Instance Attribute with Car class
class Car:
    # Attributed defined within the constructor are instance attributes
    def __init__(self, name, topspeed):
        self.name = name
        self.topspeed = topspeed
        
    def print_details(self):
        print(f"Car Name: {self.name}")
        print(f"Top Speed: {self.topspeed} Km/h")
```

```python
# Creating instances from Car class
car1 = Car("Maruti 800", 120)
car2 = Car("Ferrari", 400)
```

```python
# Printing the details of the car objects
car1.print_details()
```

```python
car2.print_details()
```

### Class Attribute
Class attributes are attributes that have the same value for all class instances. You can define a class attribute by assigning a value to a variable name outside of .__init__().<br>

**Characteristics:**
* Class attributes are shared among all instances of a class.
* They are defined outside any method within the class scope.
* Class attributes are the same for every object of the class.
* They are accessed using the class name or instance object.
* Example: className.attribute, self.attribute, etc.

```python
# Example of Class Attribute with Car class
class Car:
    # Creating a class attribute
    no_of_wheels = 4
    
    def __init__(self, name, topspeed):
        self.name = name
        self.topspeed = topspeed
        
    def print_details(self):
        print(f"Car Name: {self.name}")
        print(f"Top Speed: {self.topspeed} Km/h")
        print(f"No of Wheels: {self.no_of_wheels}")
```

```python
# Creating instances from Car class
car1 = Car("Creta", 300)
car2 = Car("Toyato", 240)
```

```python
# Printing the car details
car1.print_details()
```

```python
car2.print_details()
```

```python
# Accessing the class attributes of the Car Class
print("No of Wheels:", Car.no_of_wheels)
print("No of Wheels of car1:", car1.no_of_wheels)
print("No of Wheels of car2:", car2.no_of_wheels)
```

### Example:
Creating a Student class with Class and Instance Attributes

```python
class Student:
    school = "MBD DAV Public School"
    def __init__(self, name, rollno):
        self.name = name
        self.rollno = rollno
        
    def printDetails(self):
        print("Stuent Name:", self.name)
        print("Roll No:", self.rollno)
        print("School:", self.school)
```

```python
# Creating instances from Student class
student1 = Student("Ayan Ghosh", 1)
student2 = Student("Sayan Pal", 2)
```

```python
# Print the details of the student object
student1.printDetails()
```

```python
student2.printDetails()
```

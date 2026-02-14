[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/09 Object Oriented Programming/04_Inheritance.ipynb)

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
## Inheritance
Inheritance is a fundamental concept in object-oriented programming (OOP). It allows a class to inherit attributes and methods from another class, known as the parent or base class. Inheritance enables code reuse and promotes the concept of hierarchy and specialization.

The class which inherits the properties of the other is known as subclass (derived class or child class) and the class whose properties are inherited is known as superclass (base class, parent class).

**Example:**<br>
Let us take a real-life example to understand inheritance. Let's assume that Human is a class that has properties such as height, weight, age, etc and functionalities (or methods) such as eating(), sleeping), dreaming(), working (), etc. Now we want to create Male and Female classes. Both males and females are humans and they share some common properties (like height, weight, age, etc) and behaviours (or functionalities like eating(), sleeping(), etc), so they can inherit these properties and functionalities from the Human class. Both males and females have some characteristics specific to them (like men have short hair and females have long hair). Such properties can be added to the Male and Female classes separately. This approach makes us write less code as both the classes inherit several properties and functions from the superclass, thus we didn't have to re-write them. Also, this makes it easier to read the code.

**Python Inheritance Syntax:**<br>
```python
class SuperClass:
    # Body of base class

class SubClass(SuperClass):
    # Body of the derived class
```
<!-- #endregion -->

<center><img src="https://miro.medium.com/v2/resize:fit:812/0*PCrObyBu1ektfPWd.png" style="max-width: 650px; height: auto"></center>


### Example:
Here's an example of inheritance in Python using a superclass 'Vehicle' and subclasses 'Car' and 'Motorcycle':

```python
# Creating a superclass 'Vehicle'
class Vehicle:
    def __init__(self, brand, model, color):
        self.brand = brand
        self.model = model
        self.color = color
        
    def drive(self):
        print("Driving the vehicle.")
        
    def stop(self):
        print("Stopping the vehicle.")
```

<!-- #region -->
**Constructor in Subclass:**<br>
The constructor in subclass must call the constructor of the superclass by accessing the __init__() method of the superclass in the following format:
```python
<SuperClassName>.__init__(self, <Parameter1>, <Parameter2>, ...)
```
**Note:** The parameters being passed in this call must be same as the parameters being passed in the superclass __init__() function, otherwise it will throw an error. 
<!-- #endregion -->

```python
# Creating a subclass 'Car'
class Car(Vehicle):
    def __init__(self, brand, model, color, topSpeed):
    	super().__init__(brand, model, color)
    	self.topSpeed = topSpeed
    
    def drive(self):
        print("Driving the car.")
        
    def open_trunk(self):
        print("Opening the trunk.")
```

```python
# Creating another subclass 'Motocycle'
class Motorcycle(Vehicle):
    def __init__(self, brand, model, color, topSpeed):
        super().__init__(brand, model, color)
        self.topSpeed = topSpeed
    
    def drive(self):
        print("Driving the motorcycle")
    
    def indicator(self):
        print("Turning on the indicator.")
```

```python
# Creating instances of subclasses
car1 = Car("Maruti", "Maruti 800", "White", 150)
motorcycle1 = Motorcycle("Kawasaki", "Kawasaki Ninja H2", "Black", 293)
```

```python
# Calling overridden methods
car1.drive()
motorcycle1.drive()
```

```python
# Calling subclass-specific methods
car1.open_trunk()
motorcycle1.indicator()
```

```python
# Calling superclass methods
car1.stop()
motorcycle1.stop()
```

### Example:
Here's an example of inheritance in Python using a superclass 'Polygon' and subclasses 'Triangle':

```python
# Creating a superclass 'Polygon'
class Polygon:
    
    # Constructor
    def __init__(self, no_of_sides):
        self.no_of_sides = no_of_sides
        self.sideLengths = [0 for i in range(no_of_sides)]
        
    # Take user input for side lengths
    def inputSideLengths(self):
        self.sideLengths = [int(input("Enter Side Length: ")) for i in range(self.no_of_sides)]
        
    # Print the side lengths of the polygon
    def displaySideLengths(self):
        for i in range(len(self.sideLengths)):
            print(f"Side {i+1}: {self.sideLengths[i]}")
```

<center><img src="https://d138zd1ktt9iqe.cloudfront.net/media/seo_landing_files/area-of-triangle-with-3-sides-01-1627893596.png" style="max-width: 550px; height: auto"></center>

```python
# Creating a subclass 'Triangle'
class Triangle(Polygon):
    
    def __init__(self):
        # Calling constructor of superclas
        Polygon.__init__(self, 3)
        
    def calculateArea(self):
        a, b, c = self.sideLengths
        # Calculate the semi-perimeter
        s = (a + b + c) / 2
        area = (s * (s-a) * (s-b) * (s-c)) ** 0.5
        print("The area of the triangle is %0.2f" %area)
```

```python
# Instantiating a Triangle object
triangle1 = Triangle()

# Input the side legths of the triangle
triangle1.inputSideLengths()
```

```python
# Display the side lengths
triangle1.displaySideLengths()
```

```python
# Calculate the area of the triangle
triangle1.calculateArea()
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/09 Object Oriented Programming/07_Types_of_Methods.ipynb)

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

## Types of Methods
Inside a Python class, you can define various types of methods, each serving a specific purpose. Here are some commonly used types of methods in Python classes:


### Instance Method
Instance methods are the most common type of methods in Python classes. These are so-called instance methods because they can access the unique data of their instance. If you have two objects each created from a car class, then they each may have different properties. They may have different colors, engine sizes, seats, and so on.

Instance methods must have self as a parameter, but you don't need to pass this in every time. Self is another Python special term. Inside any instance method, you can use self to access any data or methods that may reside in your class. You won't be able to access them without going through self.

Finally, as instance methods are the most common, there's no decorator needed. Any method you create will automatically be created as an instance method unless you tell Python otherwise.

**Usage:**
* Performing calculations or operations on instance attributes.
* Modifying the state of an individual instance.
* Implementing instance-specific functionality.

```python
# Creating a Circle class with instance methods 
class Circle:
    _pi = 3.14159 # protected data member
    
    # constructor
    def __init__(self, radius):
        self.radius = radius
        
    # instance method  
    def calculate_area(self):
        area = self._pi * (self.radius)**2
        return area
    
    # instance method
    def calculate_perimeter(self):
        perimeter = 2 * self._pi * self.radius
        return perimeter
```

```python
# Creating an object from Circle class
circle1 = Circle(5)
# Calling the instance methods using the object
print("The area of the circle:", circle1.calculate_area())
print("The perimeter of the circle:", circle1.calculate_perimeter())
```

### Class Methods
Class methods are bound to the class itself rather than the instance. They have access to the class-level attributes and can be called using the class name or an instance. Class methods are defined using the **@classmethod** decorator, and the first parameter is conventionally named cls to represent the class.

**Usage:**
* Modifying or accessing class-level variables.
* Performing operations that are relevant to the class as a whole.
* Providing alternative ways to create instances of a class.

```python
# Creating a rectangle class
class Rectangle:
    width = 0 # class-level variable
    height = 0 # class-level variable
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def calculate_area(self):
        area = self.width * self.height
        return area
    
    @classmethod
    def change_size(cls, new_width, new_height):
        cls.width = new_width
        cls.height = new_height
```

```python
# Creating an instance from the rectangle class
rectangle1 = Rectangle(10, 5)
# Calling the instance method
print("The area of the rectangle:", rectangle1.calculate_area())
```

```python
# Checking the class-level variable
print(Rectangle.width)
print(Rectangle.height)
```

```python
# Modifying the class-level variable using class method
Rectangle.change_size(6, 4)
# Checking the new class-level variable
print(Rectangle.width)
print(Rectangle.height)
```

### Static Methods
Static methods, much like class methods, are methods that are bound to a class rather than its object.
They do not require a class instance creation. So, they are not dependent on the state of the object.

The difference between a static method and a class method is:
* The static method knows nothing about the class and just deals with the parameters.
* The class method works with the class since its parameter is always the class itself.

**Usage:**
* Implementing utility functions that are logically related to the class.
* Performing calculations or operations that don't require access to instance or class attributes.
* Grouping related functions together within a class for organizational purposes.

```python
# Creating a MathUtils class
class MathUtils:
    
    @staticmethod
    def add(a, b, *args):
        sum_of_numbers = a + b
        for i in args:
            sum_of_numbers += i
        return sum_of_numbers
    
    @staticmethod
    def multiply(a, b, *args):
        product_of_numbers = a * b
        for i in args:
            product_of_numbers *= i
        return product_of_numbers
```

```python
# Using the static methods
MathUtils.add(2, 3)
```

```python
MathUtils.add(2, 3, 4, 5)
```

```python
MathUtils.multiply(2, 3)
```

```python
MathUtils.multiply(2, 3, 4, 5)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Types_of_Methods.ipynb)

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


### Instance Method

```python
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
    def calulate_perimeter(self):
        perimeter = 2 * self._pi * (self.radius)
        return perimeter
```

```python
# Creating an object / instance from circle class
circle1 = Circle(5)
```

```python
circle1.radius
```

```python
print("The area of the circle:", circle1.calculate_area())
```

```python
print("The perimeter of the circle:", circle1.calulate_perimeter())
```

### Class Methods

```python
class Rectangle:
    width = 0 # class-level variable
    height = 0 # class-level variable
    
    #constructor
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    # instance method
    def calculate_area(self):
        area = self.width * self.height
        return area
        
    # class method
    @classmethod
    def change_size(cls, new_width, new_height):
        cls.width = new_width
        cls.height = new_height   
```

```python
# Creating an instance from the rectangle class
rectangle1 = Rectangle(10, 5)
```

```python
# Calling the instance method
print("The area of the rectangle is:", rectangle1.calculate_area())
```

```python
print(Rectangle.width)
print(Rectangle.height)
```

```python
print(rectangle1.width)
print(rectangle1.height)
```

```python
# Modifying the class-level variable using the class method
Rectangle.change_size(5, 5)
```

```python
# Checking the new class-level variable
print(Rectangle.width)
print(Rectangle.height)
```

### Static Methods

```python
class MathUtils:
    
    @staticmethod
    def add(x, y, *args):
        sum_of_num = x + y
        for i in args: # args is a tuple ()
            sum_of_num += i
        return sum_of_num
    
    @staticmethod
    def multiply(x, y, *args):
        product_of_num = x * y
        for i in args:
            product_of_num *= i
        return product_of_num
```

```python
MathUtils.add(10, 5)
```

```python
MathUtils.multiply(10, 5)
```

```python
MathUtils.add(10, 20, 30, 40, 17, 56) # *args = (30, 40, 17, 56)
```

```python
MathUtils.multiply(10, 20, 3, 4, 5)
```

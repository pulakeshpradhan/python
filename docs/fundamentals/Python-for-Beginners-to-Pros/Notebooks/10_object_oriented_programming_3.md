[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Notebooks/10_object_oriented_programming_3.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: py310
    language: python
    name: python3
---

# **Object Oriented Programming (Part 3)**


## **Class Relationship**

In Python, class relationships refer to how classes are related to one another and how they interact in an object-oriented programming context. Here are two primary types of relationships:

1. **Aggregation**: Represents a **"has-a"** relationship where the contained object can exist independently of the container.  

2. **Inheritance**: Represents an **"is-a"** relationship where a child class inherits properties and behaviors from a parent class.


### **Aggregation (Has-a relationship)**

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220520133409/UMLDiagram.jpg" width="60%">

```python
# Example
class Customer:
    def __init__(self, name, gender, address):
        self.name = name
        self.gender = gender
        self.address = address

    def print_address(self):
        print(f"{self.address.get_city()}, {self.address.state}-{self.address.pin}, {self.address.country}")

    def edit_profile(self, new_name, new_city, new_pin, new_state):
        self.name = new_name
        self.address.edit_addresss(new_city, new_pin, new_state)

class Address:
    def __init__(self, city, pin, state, country):
        self.__city = city # what about private attribute
        self.pin = pin
        self.state = state
        self.country = country

    def get_city(self):
        return self.__city
    
    def edit_addresss(self, new_city, new_pin, new_state):
        self.__city = new_city
        self.pin = new_pin
        self.state = new_state

address1 = Address(city="Roorkee", pin=247667, state="Haridwar", country="India")
customer1 = Customer(name="Krishnagopal Halde", gender="Male", address=address1)
customer1.print_address()

customer1.edit_profile("Akshat Goel", "Mumbai", 400001, "Maharashtra")
customer1.print_address()
```

**Brief Explanation of Aggregation in the Example:**

- **Aggregation** is demonstrated by the `Customer` class **"having an" Address** as part of its attributes (`address`).
- The `Customer` object does not directly define or manage the properties of the `Address`. Instead, it uses an independent `Address` object.
- Changes to the `Address` (via `edit_addresss`) affect the `Customer` object because the `Customer` holds a reference to the `Address` object.

**Key Points:**
1. **Independent `Address` Object:**  
   The `Address` object (`address1`) exists separately and is passed to the `Customer` constructor.

2. **Interaction with `Address`:**  
   - The `Customer` uses methods like `get_city()` and `edit_addresss()` from the `Address` class to retrieve and modify its data.
   - Modifications to the `Address` reflect automatically in the `Customer` as they share the same object.

3. **Workflow:**  
   - Initially, the address is set to `"Roorkee, Haridwar-247667, India"`.
   - After calling `edit_profile`, the `Address` object is updated to `"Mumbai, Maharashtra-400001, India"`, and `Customer` reflects this change.



### **Inheritence**

Inheritance is a fundamental concept of object-oriented programming (OOP) that allows one class (the child or derived class) to acquire the properties and behaviors of another class (the parent or base class). This enables code reuse, hierarchy creation, and easy extension of existing functionality.

<img src="https://miro.medium.com/v2/resize:fit:1400/0*5bscj-Hxw0AKkrzj.png" width="40%">

**Key Features of Inheritance:**

1. **Code Reusability:**  
   Common features can be defined in the parent class and reused in child classes.
   
2. **Hierarchy:**  
   Inheritance establishes a "is-a" relationship between classes, e.g., a `Dog` "is-a" type of `Animal`.

3. **Customization:**  
   Child classes can override or extend the methods and attributes of the parent class.

4. **Multiple and Multilevel Inheritance:**  
   Python supports:
   - **Single Inheritance:** One parent, one child.
   - **Multiple Inheritance:** One child class inherits from multiple parent classes.
   - **Multilevel Inheritance:** A child class inherits from another child class.

**Benefits of Inheritance**
- Simplifies code by reducing redundancy.
- Promotes modularity and maintainability.
- Enables polymorphism, allowing dynamic method overriding.

**Limitations**
- Overuse of inheritance can make code harder to debug and maintain.
- Alternatives like composition might be more suitable in certain scenarios.

```python
# Example
# Parent class
class User:

    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

    def login(self):
        print("login successfull!")

# Child class
class Student(User):

    def enroll(self):
        print("Enroll in the course.")

user1 = User("Krishnagopal Halder", "Male")
print("User's name:", user1.name)
print("User's gender:", user1.gender)

student1 = Student("Krishnagopal Halder", "Male")
student1.login()
print("Student's name:", student1.name)
print("Student's gender:", student1.gender)
```

#### **Class Diagram**
<img src="https://www.researchgate.net/publication/349182437/figure/fig2/AS:989911003967490@1613024583970/Class-diagram-and-inheritance.png" width="40%">


#### **What Gets Inherited?**

When a child class inherits from a parent class, the following components are inherited:

1. **Constructor**: The `__init__` method (constructor) of the parent class is inherited by the child class.
     - **Behavior:** 
       - If the child class does not define its own constructor, it will use the parent class's constructor.
       - If the child class defines its own constructor, it **overrides** the parent class's constructor.
  
2. **Non Private Attributes**: Attributes of the parent class that are not marked as private (e.g., no double underscores like `__attr`) are inherited by the child class.
    - **Behavior:** These attributes can be accessed and modified in the child class.
  
3. **Non Private Methods**: Methods of the parent class that are not private (i.e., without double underscores like `__method`) are inherited by the child class.
   - **Behavior:**
     - The child class can call these methods directly.
     - The child class can **override** these methods by redefining them.


```python
# Constructor example 1 ( If the child class does not define its own constructor)
# Parent class
class Phone:
    def __init__(self, price, brand, camera):
        self.price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print("Buying a phone")

# Child class
class Smartphone(Phone):
    pass

smartphone = Smartphone(50000, "Apple", 48)
smartphone.buy()
```

```python
# Constructor example 2 ( If the child class defines its own constructor)
# Parent class
class Phone:
    def __init__(self, price, brand, camera):
        self.price = price
        self.brand = brand
        self.camera = camera

class Smartphone(Phone):
    def __init__(self, os, ram):
        self.os = os
        self.ram = ram
        print("Inside Smartphone constructor")

smartphone = Smartphone("Android", 8)
# smartphone.brand # will throw error
```

```python
# Example 3 (Child can't access private members of the class)

class Phone:
    def __init__(self, price, brand, camera):
        print("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera
    
    # getter method
    def show(self):
        print(self.__price)

class SmartPhone(Phone):
    def check(self):
        print(self.__price)

smartphone = SmartPhone(50000, "Apple", 13)
print(smartphone.brand)
# smartphone.check() # will throw error
smartphone.show()
```

```python
# # Example 4: Guess the output
# class Parent:
#     def __init__(self, num):
#         self.__num = num

#     # getter method
#     def get_num(self):
#         return self.__num
    
# class Child(Parent):

#     def show(self):
#         print("This is in child class")

# son = Child(100)
# print(son.get_num())
# son.show()
```

```python
# # Example 5: Guess the output
# class Parent:
#     def __init__(self, num):
#         self.__num = num

#     def get_num(self):
#         return self.__num
    
# class Child(Parent):
    
#     def __init__(self, val, num):
#         self.__val = val

#     def get_val(self):
#         return self.__val
    
# son = Child(100, 10)
# print("Parent: Num:", son.get_num())
# print("Child: Val:", son.get_val())
```

```python
# # Example 6: Guess the output
# class A:
#     def __init__(self):
#         self.var1 = 100

#     def display1(self, var1):
#         print("Class A:", self.var1)

# class B(A):

#     def display2(self, var1):
#         print("Class B:", self.var1)

# obj = B()
# obj.display1(200)
```

#### **Method Overriding**

Method overriding is a feature in object-oriented programming that allows a subclass (child class) to provide a specific implementation for a method that is already defined in its superclass (parent class). The overridden method in the child class must have the same name, parameters, and return type as the method in the parent class.


```python
class Phone:
    def __init__(self, price, brand, camera):
        print("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print("Buying a phone")

class SmartPhone(Phone):
    def buy(self):
        print("Buying a smartphone")

smartphone = SmartPhone(50000, "Apple", 13)
smartphone.buy()
```

<!-- #region -->
#### **`super()` Keyword**

The `super()` keyword is used in Python to call methods or access attributes from a **parent class** (also known as the superclass) in the context of a subclass (child class). It provides a way for a child class to refer to and invoke methods or constructors from its parent class, particularly when overriding methods.


**Key Uses of `super()`:**

1. **Calling the Parent Class's Method:**  
   When a method in a child class overrides a method in a parent class, `super()` allows you to call the parent class's version of the method.

2. **Accessing Parent Class's Constructor:**  
   In a child class, `super()` can be used to call the parent class's `__init__()` constructor, enabling the child class to initialize inherited attributes.
<!-- #endregion -->

```python
class Phone:
    def __init__(self, price, brand, camera):
        print("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print("Buying a phone")


class SmartPhone(Phone):
    def buy(self):
        print("Buying a smartphone.")
        # Syntax to call the buy method of parent class
        super().buy()

smartphone = SmartPhone(50000, "Apple", 13)
smartphone.buy()
```

```python
# super -> constructor
class Phone:
    def __init__(self, price, brand, camera):
        self.price = price
        self.brand = brand
        self.camera = camera

class SmartPhone(Phone):
    def __init__(self, price, brand, camera, os, ram):
        print("Inside phone constructor")
        super().__init__(price, brand, camera)
        self.os = os
        self.ram = ram
        print("Inside smartphone constructor")

smartphone = SmartPhone(50000, "Samsung", 12, "Android", 4)

print(smartphone.os)
print(smartphone.brand)
```

#### **Inheritance in Summary**
- A class can inherit from another class.
- Inheritance improves code reuse.
- Constructor, attributes, methods get inherited to the child class.
- The parent has no access to the child class.
- Private properties of parent are not accessible directly in child class.
- Child class can override the attributes or methods. This is called method overriding.
- `super()` is an inbuilt function which is used to invoke the parent class methods and constructor.

```python
# # Guess the output
# class Parent:
#     def __init__(self, num):
#         self.__num = num

#     def get_num(self):
#         return self.__num
    
# class Child(Parent):
#     def __init__(self, num, val):
#         super().__init__(num)
#         self.__val = val

#     def get_val(self):
#         return self.__val
    
# son = Child(100, 200)
# print(son.get_num())
# print(son.get_val())
```

```python
# # Guess the output
# class Parent:
#     def __init__(self):
#         self.num = 100

# class Child(Parent):
#     def __init__(self):
#         super().__init__()
#         self.var = 200

#     def show(self):
#         print(self.num)
#         print(self.var)

# son = Child()
# son.show()
```

```python
# # Guess the output
# class Parent:
#     def __init__(self):
#         self.__num = 100

#     def show(self):
#         print("Parent:", self.__num)

# class Child(Parent):
#     def __init__(self):
#         super().__init__()
#         self.__var = 10

#     def show(self):
#         print("Child:", self.__var)

# obj = Child()
# obj.show()
```

```python
# # Guess the output
# class Parent:
#     def __init__(self):
#         self.__num = 100

#     def show(self):
#         print("Parent:", self.__num)

# class Child(Parent):
#     def __init__(self):
#         super().__init__()
#         self.__var = 10

#     def show(self):
#         print("Child:", self.__var)


# obj = Child()
# obj.show()
```

<!-- #region -->
#### **Types of Inheritance**
<img src="https://i.pinimg.com/originals/4e/d9/b2/4ed9b2b640e3e64ed2ae4fbf6a480c75.jpg" width="40%">

Inheritance in Python allows a class (child class) to derive properties and behaviors from another class (parent class). Python supports several types of inheritance:

**1. Single Inheritance**
- **Definition:** A single child class inherits from a single parent class.
- **Usage:** Simplifies code reuse and enhances modularity.


**2. Multiple Inheritance**
- **Definition:** A child class inherits from more than one parent class.
- **Usage:** Useful when a class needs to inherit functionality from multiple classes.
- **Method Resolution Order (MRO):** Determines the order in which methods are called in the inheritance hierarchy.


**3. Multilevel Inheritance**
- **Definition:** A child class inherits from a parent class, and another child class inherits from this child class.
- **Usage:** Creates a chain of inheritance.

**4. Hierarchical Inheritance**
- **Definition:** Multiple child classes inherit from a single parent class.
- **Usage:** Useful for creating subclasses with shared base functionality.

**5. Hybrid Inheritance**
- **Definition:** A combination of two or more types of inheritance (e.g., multiple and multilevel inheritance).
- **Usage:** Allows for flexible and complex hierarchies.


**Summary Table**

| **Type of Inheritance** | **Description**                           | **Example**             |
|--------------------------|-------------------------------------------|-------------------------|
| Single                  | One parent, one child                     | `A → B`                |
| Multiple                | Multiple parents, one child               | `A, B → C`             |
| Multilevel              | Chain of inheritance                      | `A → B → C`            |
| Hierarchical            | One parent, multiple children             | `A → B, C`             |
| Hybrid                  | Combination of multiple inheritance types | Complex combinations   |

Each type of inheritance is suited for specific use cases and offers various levels of code reuse and modularity.
<!-- #endregion -->

```python
# Single inheritance
class Phone:
    def __init__(self, price, brand, camera):
        self.price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print("Buying a phone")

class SmartPhone(Phone):
    pass

SmartPhone(50000, "Apple", 13).buy()
```

```python
# Multilevel inheritance
class Product:
    def review(self):
        print("Product customer review")

class Phone(Product):
    def __init__(self, price, brand, camera):
        print("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print("Buying a phone")

class SmartPhone(Phone):
    pass

smartphone = SmartPhone(50000, "Apple", 13)
smartphone.buy()
smartphone.review()
```

```python
# Hierarchical inheritance
class Phone:
    def __init__(self, price, brand, camera):
        self.__price = price
        self.brand = brand
        self.camera = camera
    
    def buy(self):
        print("Buying a phone.")

class SmartPhone(Phone):
    pass

class FeaturePhone(Phone):
    pass

SmartPhone(50000, "Apple", 13).buy()
FeaturePhone(5000, "Nokia", 2).buy()
```

```python
# Multiple inheritance
class Phone:
    def __init__(self, price, brand, camera):
        print("Inside price constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print("Buying a phone")

class Product:
    def review(self):
        print("Customer review")

class SmartPhone(Phone, Product):
    pass

smartphone = SmartPhone(50000, "Apple", 13)
smartphone.buy()
smartphone.review()
```

<!-- #region -->
**The Diamond Problem**

The **diamond problem** arises in **multiple inheritance**, where a class inherits from two classes that both inherit from a common base class. It creates ambiguity about which path should be followed when invoking methods or accessing attributes from the common base class.


**Why is it Called the Diamond Problem?**
The inheritance hierarchy forms a diamond shape:

```
      A
     / \
    B   C
     \ /
      D
```

- **Class A:** The common base class.
- **Classes B and C:** Intermediate classes that inherit from A.
- **Class D:** The child class that inherits from both B and C.

When a method in `D` calls a method or accesses an attribute from `A`, it's ambiguous whether the call should go through `B` or `C`.


**Method Resolution Order (MRO)**

To resolve the diamond problem, Python uses the **Method Resolution Order (MRO)**. The MRO determines the order in which classes are searched for methods or attributes.

Python uses the **C3 linearization algorithm** (also called **C3 superclass linearization**) to compute the MRO. The MRO ensures:
1. **Consistency**: A method is called from the first valid class found in the order.
2. **No Ambiguity**: Python resolves the order of calls without ambiguity.
3. **Breadth-First Resolution**: The child class is searched first, followed by its parents (left to right), and then their parents.
<!-- #endregion -->

```python
# The diamond problem
class Phone:
    def __init__(self, price, brand, camera):
        print("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print("Buying a phone")

class Product:
    def buy(self):
        print("Product buy method")

# Method resolution order
class SmartPhone(Product, Phone):
    pass

smartphone = SmartPhone(50000, "Apple", 13)
smartphone.buy()
```

```python
# # Guess the output
# class A:
#     def m1(self):
#         return 20
    
# class B(A):
#     def m1(self):
#         return 30
#     def m2(self):
#         return 40
    
# class C(B):
#     def m2(self):
#         return 20
    
# obj1 = A()
# obj2 = B()
# obj3 = C()
# print(obj1.m1() + obj3.m1() + obj3.m2())
```

```python
# # Guess the output
# class A:
#     def m1(self):
#         return 20
    
# class B(A):
#     def m1(self):
#         val = super().m1()+30
#         return val
    
# class C(B):
#     def m1(self):
#         val = self.m1()+20
#         return val
    
# obj = C()
# print(obj.m1())
```

<!-- #region -->
## **Polymorphism**

**Polymorphism** is a concept in object-oriented programming that allows objects of different classes to be treated as objects of a common superclass. It enables a single interface (method or function) to operate on different types of objects.


**Key Points:**
1. **"Many forms":** The term "polymorphism" means "many forms." A single method or function behaves differently based on the object or class it is acting upon.
2. **Dynamic Behavior:** The actual implementation executed is determined at runtime, making Python a dynamically-typed and polymorphic language.



**Types of Polymorphism**

**1. Method Overriding (Runtime Polymorphism)**
- **Definition:** A subclass provides a specific implementation for a method already defined in its parent class.

**2. Method Overloading (Static Polymorphism)**
- **Definition:** A method in a class is defined with different parameter configurations. However, Python doesn't support true method overloading. It can be emulated using default arguments or variable-length arguments.

**3. Operator Overloading**
- **Definition:** Operators like `+`, `*`, etc., behave differently depending on the operands. This is implemented using special methods (dunder methods).

**4. Polymorphism with Functions and Objects**
- **Definition:** A single function can operate on objects of different classes, provided they share a common interface.

**Advantages of Polymorphism**
1. **Code Reusability:** Same interface can work with different data types or classes.
2. **Flexibility:** New functionality can be added without altering existing code.
3. **Extensibility:** Objects can evolve while maintaining the same interface.


**Summary Table**

| **Type**                  | **Example**                                 |
|---------------------------|---------------------------------------------|
| **Method Overriding**      | Subclasses redefining parent class methods. |
| **Method Overloading**     | Same method name with different parameters (emulated in Python). |
| **Operator Overloading**   | Operators working differently based on operand type. |
| **Polymorphism with Functions** | Functions accepting objects of different classes. |

Polymorphism enables flexibility and dynamic behavior in Python, making it a core concept in object-oriented programming.
<!-- #endregion -->

```python
# # Method overloading
# class Shape:

#     def area(self, radius):
#         return 3.14*radius*radius
    
#     def area(self, l, b):
#         return l*b
    
# s = Shape()

# s.shape(2)
# s.area(3, 4)
```

```python
# Alternative way of method overloading in python
class Shape:
    def area(self, a, b=0):
        if b == 0:
            return 3.14*a*a
        else:
            return a*b
        
s = Shape()

print(s.area(2))
print(s.area(3, 4))
```

```python
# Operator overloading
"Hello" + " World" # '+' operator is used for concatenation
```

```python
4 + 5 # '+' operator is used for addition
```

```python
[1, 2, 3] + [4, 5] # '+' operator is used for merginge
```

<!-- #region -->
## **Abstraction**

**Abstraction** is an object-oriented programming (OOP) concept that focuses on **hiding the implementation details** of a feature while exposing only the necessary functionalities. It allows developers to work with high-level interfaces and simplifies the process of designing and using complex systems.

**Key Characteristics of Abstraction**

1. **Hiding Details:**
   - The user doesn't need to know how the functionality is implemented; they only interact with the interface.
   - This reduces complexity and enhances security by restricting access to sensitive code.

2. **High-Level Interfaces:**
   - Only the essential features of an object are shown, while the internal implementation is hidden.

3. **Implemented Using Abstract Classes and Interfaces:**
   - Abstract classes are blueprints for other classes.
   - They cannot be instantiated directly and are meant to be inherited.


**How Abstraction Works in Python**

Python provides abstraction through the use of:
1. **Abstract Base Classes (ABCs):**
   - Defined using the `abc` module.
   - Classes with one or more abstract methods are abstract classes.
2. **Abstract Methods:**
   - Methods declared but not implemented in an abstract class.

**Advantages of Abstraction**

1. **Reduces Complexity:**
   - Users interact with the essential features without worrying about the implementation.
2. **Improves Code Reusability:**
   - Abstract classes and methods can be reused across different projects.
3. **Promotes Flexibility:**
   - Allows developers to change implementations without altering the interface.
4. **Enhances Security:**
   - Hides sensitive or unnecessary details from the end user.

**Summary**

| **Concept**                | **Description**                                          |
|----------------------------|----------------------------------------------------------|
| **Abstraction**             | Hiding implementation details and exposing functionalities. |
| **Abstract Class**          | A class with abstract methods that acts as a blueprint.  |
| **Abstract Method**         | A method declared in an abstract class but not implemented. |
| **Concrete Class**          | A class that implements the abstract methods.           |

**Abstraction** helps to design robust systems by defining clear interfaces, hiding implementation details, and enabling developers to focus on high-level interactions.
<!-- #endregion -->

```python
from abc import ABC, abstractmethod

class BankApp(ABC):
    def database(self):
        print("Connected to database")

    @abstractmethod
    def security(self):
        pass

    @abstractmethod
    def display(self):
        pass
```

```python
class MobileApp(BankApp):

    def mobile_login(self):
        print("login into mobile")

    def security(self):
        print("mobile security")

    def display(self):
        print("display")
```

```python
mob = MobileApp()
```

```python
mob.security()
mob.display()
```

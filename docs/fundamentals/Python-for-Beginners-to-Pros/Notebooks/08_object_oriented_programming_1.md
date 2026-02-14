[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Notebooks/08_object_oriented_programming_1.ipynb)

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

# **Object Oriented Programming (Part 1)**

Object-Oriented Programming (OOP) is a programming paradigm based on the concept of "objects," which can contain both data (in the form of fields, also known as attributes or properties) and code (in the form of methods, which are functions). OOP allows for the structuring of software in a way that models real-world entities, making the code more modular, flexible, and easier to maintain.

In Object-Oriented Programming (OOP), we can build our own data types using **classes**. These user-defined data types (also known as **custom classes**) allow us to create objects that represent real-world entities or abstract concepts, with their own attributes (data) and methods (functions).

For example, instead of relying solely on primitive data types like integers or strings, we can design a class to model more complex data, such as a `Person`, `Car`, or `BankAccount`. Each of these classes can encapsulate specific properties (e.g., name, balance, speed) and behaviors (e.g., withdraw money, accelerate, display information), enabling us to work with these objects in a structured and intuitive way.

By defining our own data types, we extend the language’s capabilities and create reusable, modular components that can fit the needs of our particular application. This flexibility is one of the core strengths of OOP, promoting better code organization, readability, and maintainability.

<img src="https://lh6.googleusercontent.com/yPsibbUh1aHOvi0U3-wtdlNpWWutbyYULv1PLkx0QlOOq81DiXVvPgvKVrtY7Ef1yZF5NLabXrHBjHL80lx9hTqR_64jGRFZdbR9FIs4LDR9RcEn1M9LX_D5i4fYKR4vNZA-dZ9R" width="50%">



## **Class and Object**

<img src="https://blog.kakaocdn.net/dn/cRfL7j/btrgJYeGDQP/r7GcNfnwCfJoBJwK4uhBw1/img.png" width="50%">

In Object-Oriented Programming (OOP), **Class** and **Object** are fundamental concepts.

**Class:** A **class** is a blueprint or template for creating objects. It defines a set of attributes (data) and methods (functions) that the objects created from the class will have.

**Object:** An **object** is an instance of a class. When a class is defined, no memory is allocated until we create objects. Each object has its own values for the attributes defined in the class and can use the methods of the class.

For example, when you work with Python’s built-in data types like lists, strings, or integers, you’re actually working with objects of classes predefined in Python. These objects have their own methods and properties.

Let’s analyze the following code:

```python
L = [1, 2, 3]
print(type(L))  # Output: <class 'list'>
# L.upper()       # This will raise an AttributeError
```


- `L = [1, 2, 3]`: Here, `L` is an **object** of the class `list`. The class `list` is the blueprint for Python’s list data structure, and `L` is an instance (or object) of this class.
- `print(type(L))`: This statement prints the type of `L`, which is `<class 'list'>`. This tells us that `L` is an object of the class `list`.
- `L.upper()`: This will raise an `AttributeError` because the `list` class does not have a method called `upper`. The `upper()` method exists for `str` (string) objects, not for lists.



## **Creating Class and Object**

```python
# We should write class name in Pascal Case
# Write a class to represent an ATM system
class ATM:
    # constructor (special function)
    def __init__(self):
        self.pin = ""
        self.balance = 0
        self.menu()

    def menu(self):
        user_input = input(
            """
            Hi how can I help you?
            1. Press 1 to create pin
            2. Press 2 to change pin
            3. Press 3 to check balance
            4. Press 4 to withdraw
            5. Anything else to exit
            """)

        if user_input == "1":
            self.create_pin()

        elif user_input == "2":
            self.change_pin()

        elif user_input == "3":
            self.check_balance()

        elif user_input == "4":
            self.withdraw()

        else:
            exit()
        
    def create_pin(self):
        user_pin = input("Enter your PIN:")
        self.pin = user_pin

        user_balance = int(input("Enter balance:"))
        self.balance = user_balance

        print("PIN created successfully!")
        self.menu()

    def change_pin(self):
        old_pin = input("Enter old PIN:")

        if old_pin == self.pin:
            # Let user change the PIN
            new_pin = input("Enter new PIN:")
            self.pin = new_pin
            print("PIN change successfully.")
            self.menu()

        else:
            print("You entered incorrect PIN.")
            self.menu()

    def check_balance(self):
        user_pin = input("Enter your PIN:")
        if user_pin == self.pin:
            print("Your balance is ", self.balance)

        else:
            print("You entered incorrect PIN.")

    def withdraw(self):
        user_pin = input("Enter your PIN:")
        if user_pin == self.pin:
            amount = int(input("Enter the amount"))
            if amount <= self.balance:
                self.balance = self.balance - amount
                print("Withdrawl successfull. New balance is ", self.balance)

            else:
                print("Your account has insufficient balance.")

        else:
            print("You entered incorrect PIN.")

        self.menu()
```

```python
# Create an object of the ATM class
# obj = ATM()
# print(type(obj))
```

## **Class Diagram**
Here’s the class diagram for the given `ATM` system, following the conventions where class names are written in **Pascal Case**:

**ATM Class Diagram**

```
+---------------------+
|        ATM          |
+---------------------+
| - pin: str          |
| - balance: int      |
+---------------------+
| + __init__()        |
| + menu()            |
| + create_pin()      |
| + change_pin()      |
| + check_balance()   |
| + withdraw()        |
+---------------------+
```

**Explanation:**

- **ATM** is the class name.
- **Attributes (fields)**:
  - `pin`: A private attribute that stores the user's PIN. It's initialized as an empty string.
  - `balance`: A private attribute that stores the user's account balance. It's initialized as `0`.
  
- **Methods (operations)**:
  - `__init__()`: The constructor, which initializes the `pin` and `balance` attributes and calls the `menu()` method.
  - `menu()`: Displays a menu with options for creating a PIN, changing a PIN, checking balance, withdrawing money, or exiting the program.
  - `create_pin()`: Allows the user to create a PIN and set an initial balance.
  - `change_pin()`: Allows the user to change the current PIN if the old PIN is correctly provided.
  - `check_balance()`: Prompts the user to enter their PIN and displays the account balance if the PIN is correct.
  - `withdraw()`: Prompts the user for their PIN and withdraws the specified amount from the account if the PIN is correct and there are sufficient funds.

In class diagrams, **"+"** and **"-"** symbols represent the visibility (or access control) of the class members (attributes and methods). Here's what they mean:

- **"+" (Public):**
  - This indicates that the attribute or method is **public**, meaning it can be accessed from outside the class. In Python, all attributes and methods are public by default unless otherwise specified.

- **"-" (Private):**
  - This indicates that the attribute or method is **private**, meaning it is only accessible from within the class itself and not from outside the class. In Python, you can make an attribute or method private by using a leading double underscore (`__`), though it's more of a convention in Python as the language doesn't enforce strict private access.


## **Methods vs Functions**
In Python, the terms **methods** and **functions** are often used interchangeably, but they have distinct meanings based on their context.

1. **Functions:**
   - A **function** is a block of reusable code that performs a specific task. It can exist independently and is not associated with any particular object.
   - Functions are defined using the `def` keyword and can take inputs (parameters) and return outputs (values).

2. **Methods:**
   - A **method** is similar to a function but is associated with an object (an instance of a class). In other words, a method is a function that "belongs to" an object.
   - Methods are called on objects and can access and modify the data (attributes) within the object they belong to.
   - Methods are defined inside a class and must take at least one parameter (`self`), which refers to the instance of the class.

```python
L = [1, 2, 3]
len(L) # function -> because it is outside the 'list' class
L.append(4) # method -> because it is inside the 'list' class
```

## **Magic Methods (a.k.a Dunder Methods)**
**Magic methods** in Python, also known as **dunder (double underscore) methods**, are special methods that allow you to define how objects of a class should behave in certain operations. They have names that begin and end with double underscores (`__`), such as `__init__`, `__str__`, `__add__`, etc.

Magic methods are used to implement the behavior of operators, object construction, type conversion, and other operations. Python automatically invokes these methods in specific situations, making them very powerful for customizing the behavior of objects.

**Common Magic Methods:**

1. **`__init__(self, ...)`** (Constructor)
   - Called when a new instance of a class is created. It is used to initialize the object's state.

2. **`__str__(self)`** (String Representation)
   - Called when `str()` or `print()` is used on an object. It defines the human-readable string representation of the object.

3. **`__len__(self)`** (Length of Object)
   - Called when `len()` is used on an object. It defines how the object should respond when its length is queried.


## **Concept of `self`**
In Python, the **`self`** keyword is used in the context of **class** and **instance methods**. It refers to the current instance of the class and allows access to the instance's attributes and methods.

**Characteristics:**
- `self` is a reference to the current instance of the class.
- It is used to access attributes and methods associated with that instance.
- `self` is passed automatically to instance methods in Python.
- It’s a convention to name the first parameter of instance methods `self`, but any name can technically be used (though not recommended).

```python
# Write a class to represent an ATM system
class ATM:
    def __init__(self):
        self.pin = ""
        self.balance = 0
        print("Address of self:", id(self))
        # self.menu()

    def menu(self):
        user_input = input(
            """
            Hi how can I help you?
            1. Press 1 to create pin
            2. Press 2 to change pin
            3. Press 3 to check balance
            4. Press 4 to withdraw
            5. Anything else to exit
            """)

        if user_input == "1":
            self.create_pin()
        else:
            exit()
        
    def create_pin(self):
        user_pin = input("Enter your PIN:")
        self.pin = user_pin

        user_balance = int(input("Enter balance:"))
        self.balance = user_balance

        print("PIN created successfully!")
        self.menu()
```

```python
# Create an object of the ATM class
obj = ATM()
print("Address of the object:", id(obj))
```

```python
# Create another object of the ATM class
obj2 = ATM()
print("Address of the object 2:", id(obj2))
```

```python
# class Fraction
```

## **Build Custom Class**

```python
# Create a 'Fraction' class to represent fractions and perform arithmetic operations
class Fraction:

    # Parameterized Constructor: Initializes the numerator and denominator
    def __init__(self, x, y):
        self.num = x  # Numerator of the fraction
        self.den = y  # Denominator of the fraction

    # String representation of the fraction (e.g., '3/4')
    def __str__(self):
        return "{}/{}".format(self.num, self.den)

    # Addition of two fractions
    def __add__(self, other):
        new_num = self.num * other.den + other.num * self.den
        new_den = self.den * other.den

        return "{}/{}".format(new_num, new_den)

    # Subtraction of two fractions
    def __sub__(self, other):
        new_num = self.num * other.den - other.num * self.den
        new_den = self.den * other.den

        return "{}/{}".format(new_num, new_den)

    # Multiplication of two fractions
    def __mul__(self, other):
        new_num = self.num * other.num
        new_den = self.den * other.den

        return "{}/{}".format(new_num, new_den)

    # Division of two fractions
    def __truediv__(self, other):
        new_num = self.num * other.den
        new_den = self.den * other.num

        return "{}/{}".format(new_num, new_den)

    # Convert the fraction to decimal
    def convert_to_decimals(self):
        return self.num / self.den
```

```python
# Create two fraction objects
fraction1 = Fraction(2, 4)
fraction2 = Fraction(4, 6)

print("Fraction 1 object:", fraction1)
print("Fraction 2 object:", fraction2)
print("Addition of two fractions:", fraction1 + fraction2)
print("Subtraction of two fractions:", fraction1 - fraction2)
print("Multiplication of two fractions:", fraction1 * fraction2)
print("Division of two fractions:", fraction1 / fraction2)
print("Decimal representation of Fraction 1:", fraction1.convert_to_decimals())
print("Decimal representation of Fraction 2:", fraction2.convert_to_decimals())
```

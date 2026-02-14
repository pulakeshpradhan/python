[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Notebooks/09_object_oriented_programming_2.ipynb)

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

# **Object Oriented Programming (Part 2)**

<!-- #region vscode={"languageId": "plaintext"} -->
## **Problem Statement:** Write OOP classes to handle the following scenarios:
- A user can create and view 2D coordinates
- A user can find out the distance between 2 coordinates
- A user can find the distance of a coordinate from origin
- A user can check if a point lies on a given line
- A user can find the distance between a given 2D point and a given line
<!-- #endregion -->

```python
# Write classes for coordinate geometry
class Point:

    # parameterized constructor
    def __init__(self, x, y):
        self.x_coord = x
        self.y_coord = y

    # print representation using magic methods
    def __str__(self):
        return "<{}, {}>".format(self.x_coord, self.y_coord)
    
    # method to calculate euclidean distance
    def euclidean_distance(self, other):
        return ((self.x_coord - other.x_coord)**2 + (self.y_coord - other.y_coord)**2)**0.5
    
    # method to distance from origin
    def distance_from_origin(self):
        # return (self.x_coord**2 + self.y_coord*82)**5 # alternative
        return self.euclidean_distance(Point(0, 0))
    
class Line:

    # parameterized constructor
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

    # print representation using magic methods
    def __str__(self):
        return f"{self.A}x + {self.B}y + {self.C} = 0"
    
    # method to check a point fall on a line
    def point_on_line(line, point):
        if line.A*point.x_coord + line.B*point.y_coord + line.C == 0:
            return True # point falls on the line
        else:
            return False # point doesn't fall on the line
    
    # method to calculate shortest distance between a line and a point
    def shortest_distance(line, point):
        return abs(line.A*point.x_coord + line.B*point.y_coord + line.C) / (line.A**2 + line.B**2)**0.5
    
    # method to check whether two line segments intersect each other
    def is_intersected(line1, line2):
        if ((line1.A / line2.A) != (line1.B / line2.B)) or ((line1.A / line2.A) == (line1.B / line2.B) == (line1.C / line2.C)):
            return True # lines are intersecting each other
        else:
            return False # lines are not intersecting each other
```

```python
# Create two Point objects
point1 = Point(0, 0)
point2 = Point(1, 1)

# Create three Line objects
line1 = Line(1, 1, -2)
line2 = Line(2, 1, 1)
line3 = Line(1, 1, -4)

print("First point object:", point1)
print("Second point object:", point2)
print("First line object:", line1)

print("Euclidean distance between point1 and point2", point1.euclidean_distance(point2))
print("Distance of the point2 from the origin", point2.distance_from_origin())
print("point2 falls on the line1?:", line1.point_on_line(point2))
print("Shortest distance between line1 and point1 is:", line1.shortest_distance(point1))
print("Line2 is intersecting Line3:", line2.is_intersected(line3))
```

## **How Object access Attributes**
In Python, objects access attributes using the dot notation (`object.attribute`).

```python
# Create a Person class
class Person:

    def __init__(self, name, country):
        self.name = name
        self.country = country

    def greet(self):
        if self.country == "india":
            print("Namaste", self.name)
        else:
            print("Hello", self.name)
```

```python
# Create a Person object
person1 = Person("Krishnagopal", "India")

# accessing the attributes
print("Name:", person1.name)
print("Country:", person1.country)

# accessing the methods
person1.greet()
```

## **Attribute Creation from Outside of the Class**

```python
# Create a new attribute to the object outside of the class
person1.gender = "male"
person1.gender
```

## **Reference Variables**
A **reference variable** in Python is a variable that refers to (or "points to") an object in memory. Python variables do not hold the actual data directly but instead hold a reference (memory address) to the object where the data is stored.

- Reference variables hold the objects
- We can create objects without reference variable as well
- An object can have multiple reference variables
- Assigning a new reference variable to an existing object does not create a new object

```python
# Object without a reference
class Person:

    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

Person("krishnagopal", "male")
```

```python
p = Person("krishnagopal", "male")
q = p

# multiple reference
print(id(p))
print(id(q))
```

```python
print(p.name)
print(q.name)
q.name = "ankit"
print(q.name)
print(p.name)
```

## **Pass by Reference**
In programming, pass by reference refers to passing the memory address of a variable (a reference) to a function. This means that if the function modifies the parameter, it directly affects the original variable since both refer to the same memory location.

```python
class Person:

    def __init__(self, name, gender):
        self.name = name
        self.gender = gender
    
# outside the class -> function
def greet(person):
    print(f"Hi! my name is {person.name} and I am a {person.gender}.")
    p1 = Person("ankit", "male")
    return p1

p = Person("krishnagopal", "male")
x = greet(p)
print(x.name)
print(x.gender)
```

## **Mutability of an Object**

In Python, **mutability** refers to whether an object's value can be changed after it has been created. Objects in Python fall into two categories based on their mutability:

1. **Mutable Objects**: Can be changed after creation.
2. **Immutable Objects**: Cannot be changed after creation.

To determine whether an object is mutable or immutable, you can check the memory address of the object using `id()`. If the address changes after modification, the object is immutable.

```python
class Person:

    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

# outside of the class -> function
def greet(person):
    person.name = "ankit"
    return person

p = Person("krishnagopal", "male")
print(id(p))
p1 = greet(p)
print(id(p1))
```

## **Instance Variable**
An instance variable is a variable that is tied to a specific instance of a class. Each object (or instance) of a class can have its own unique values for these variables. These variables store the state or attributes of the object.

```python
# instance variable -> name, country
class Person:

    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

p1 = Person("ankit", "india")
p2 = Person("krishnagopal", "germany")
```

```python
print(p1.name)
print(p2.name)
```

<!-- #region -->
## **Encapsulation**

Encapsulation in Python is a fundamental concept in object-oriented programming (OOP) that refers to bundling data (attributes) and the methods (functions) that operate on the data into a single unit, typically a class. Encapsulation helps in restricting direct access to certain components of an object, thereby maintaining the integrity of the data and promoting modular, secure code design.


**Key Aspects of Encapsulation:**
1. **Data Hiding**:
   - Encapsulation provides a way to hide the internal state of an object and protect it from unintended interference and misuse.
   - Access to the object's internal state is controlled through public methods (getters and setters).

2. **Access Modifiers**:
   - **Public**: Members (attributes or methods) with no underscores are accessible from anywhere.
   - **Protected**: Members prefixed with a single underscore (`_attribute`) indicate that they are intended to be used within the class or subclasses (not enforced by Python but a convention).
   - **Private**: Members prefixed with a double underscore (`__attribute`) are name-mangled to prevent direct access outside the class.

**Advantages of Encapsulation:**
- **Improves Security**: Protects sensitive data from being accessed or modified accidentally.
- **Enhances Code Modularity**: Changes to the internal implementation of a class can be made without affecting the external code.
- **Promotes Maintainability**: Encapsulation simplifies debugging and updating of code.
- **Encourages Abstraction**: Focuses on what an object does rather than how it does it.

Encapsulation is an essential part of designing robust and scalable systems in Python.
<!-- #endregion -->

```python
# Create an ATM class
class ATM:

    def __init__(self):
        self.pin = ""
        self.__balance = 0
        self.menu()

    def get_balance(self):
        return self.__balance
    
    def set_balance(self, new_balance):
        if type(new_balance) == int:
            self.__balance = new_balance

        else:
            print("Only integer is supported")

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
            pass

    def create_pin(self):
        user_pin = input("Enter your PIN: ")
        self.pin = user_pin

        user_balance = input("Enter your balance: ")
        self.__balance = user_balance

        print("PIN created successfully")

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
            print("Your balance is ", self.__balance)

        else:
            print("You entered incorrect PIN.")

    def withdraw(self):
        user_pin = input("Enter your PIN:")
        if user_pin == self.pin:
            amount = int(input("Enter the amount"))
            if amount <= self.__balance:
                self.__balance = self.__balance - amount
                print("Withdrawl successfull. New balance is ", self.__balance)

            else:
                print("Your account has insufficient balance.")

        else:
            print("You entered incorrect PIN.")

        self.menu()
```

```python
# atm_obj = ATM()
# atm_obj.get_balance()
```

```python
# atm_obj.set_balance(1000)
# atm_obj.get_balance()
```

```python
# atm_obj.withdraw()
```

## **Collection of Objects**

In Python, objects can be organized into collections like lists, sets, or dictionaries, enabling efficient storage and manipulation of related data. For example, by creating a list of objects from a class, we can store multiple instances and process them collectively. This allows operations like iteration, filtering, or accessing specific attributes for all objects in the collection. This approach is particularly useful for managing and performing batch operations on similar data entities.

```python
# List of objects
class Person:

    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

p1 = Person("krishnagopal", "male")
p2 = Person("abir", "male")
p3 = Person("dipak", "male")

L = [p1, p2, p3]

for i in L:
    print(i.name)
```

```python
D = {"person1": p1, "person2": p2, "person3": p3}

for key, value in D.items():
    print(key, ":", value.name, ",", value.gender)
```

## **Static Variables**
A **static variable** in Python refers to a variable that is shared among all instances of a class. It belongs to the class rather than any specific instance, meaning its value is the same across all objects of that class unless explicitly modified. Static variables are defined at the class level, outside of any instance methods.

**Characteristics of Static Variables:**
1. **Class-Level Scope**:
   - Static variables are declared inside the class but outside any instance methods or constructors.
   - They are shared across all instances of the class.

2. **Shared State**:
   - All instances of the class share the same static variable, and any changes to it are reflected across all instances.

3. **Access**:
   - Can be accessed using the class name (`ClassName.variable`) or an instance (`object.variable`), though the former is preferred for clarity.

**Key Points to Remember:**
1. **Static vs Instance Variables**:
   - **Static variables** are shared across all objects of a class.
   - **Instance variables** are unique to each object and defined within methods using `self`.

2. **Use Cases**:
   - Storing values that should remain consistent across all instances, like configuration settings or counters.
   - Tracking shared state or data across all instances.

```python
# Write a class for ATM
class ATM:

    __counter = 1

    def __init__(self):
        self.__pin = ""
        self.__balance = 0
        self.cid = 0
        self.cid = ATM.__counter
        ATM.__counter = ATM.__counter + 1

        self.menu()

    # utility functions
    @staticmethod
    def get_counter():
        return ATM.__counter
    
    def __str__(self):
        print("ATM_instance")

    def get_pin(self):
        return self.__pin
        
    def get_balance(self):
        return self.__balance
    
    def set_balance(self, new_balance):
        self.__balance = new_balance
        return self.__balance
    
    def get_cid(self):
        return self.cid
    

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
            pass

    def create_pin(self):
        user_pin = int(input("Enter your PIN: "))
        self.__pin = user_pin
        print("PIN created successfully!")

        user_balance = int(input("Enter your balance: "))
        self.__balance = user_balance
        print("Balance stored successfully!")

        self.menu()


    def change_pin(self):
        new_pin = int(input("Enter old PIN: "))
        if new_pin == self.__pin:
            self.__pin = new_pin
            print("PIN changed successfully!")

        else:
            print("You entered the wrong PIN.")

        self.menu()

    def check_balance(self):
        user_pin = int(input("Enter old PIN: "))
        if user_pin == self.__pin:
            print("Your account balance is:", self.__balance)

        else:
            print("You entered the wrong PIN.")

        self.menu()

    def withdrawl(self):
        user_pin = int(input("Enter old PIN: "))
        if user_pin == self.__pin:
            amount = int(input("Enter the amount: "))
            self.__balance = self.__balance - amount
            print("Your new balance is:", self.__balance)

        else:
            print("You entered the wrong PIN.")

        self.menu()
```

```python
# Create an ATM object
p1 = ATM()

# Print the counter id
print("Counter ID of the object:", p1.get_cid())
```

```python
# Create another ATM object
p2 = ATM()

# Print the counter id
print("Counter ID of the object:", p2.get_cid())
```

```python
# Call the static method with the class name
ATM.get_counter()
```

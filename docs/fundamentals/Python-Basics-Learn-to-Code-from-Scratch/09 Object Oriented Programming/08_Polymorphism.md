[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/09 Object Oriented Programming/08_Polymorphism.ipynb)

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

## Polymorphism
The literal meaning of polymorphism is the condition of occurrence in different forms. Polymorphism is a very important concept in programming. It refers to the use of a single type entity (method, operator, or object) to represent different types in different scenarios. Let's take a few examples:


#### Example 1: Polymorphism in addition(+) operator
We know that the + operator is used extensively in Python programs. But, it does not have a single usage. For integer data types, the + operator is used to perform an arithmetic addition operation.

```python
a = 5
b = 10
print(a+b)
```

Hence, the above program outputs 15. <br>
Similarly, for string data types, the + operator is used to perform concatenation.

```python
str1 = "Geo"
str2 = "Next"
print(str1 + str2)
```

As a result, the above program outputs "GeoNext". <br>
Here, we can see that a single operator + has been used to carry out different operations for distinct data tvpes. This is one of the most simple occurrences of polymorphism in Python.


#### Example 2: Functional Polymorphism in Python
There are some functions in Python which are compatible to run with multiple data types.
One such function is the len() function. It can run with many data types in Python. Let's look at some example use cases of the function:

```python
print(len("GeoNext"))
print(len([1, 2, 4, 2, 7]))
print(len({"a": 1, "b": 2}))
```

Here, we can see that many data types such as string, list, tuple, set, and dictionary can work with the len() function. However, we can see that it returns specific information(the length) about the specific data types.


### Class Polymorphism in Python
Polymorphism is a very important concept in Object-Oriented Programming. We can use the concept of polymorphism while creating class methods as Python allows different classes to have methods with the same name.

We can then later generalize calling these methods by disregarding the object we are working with.
Let's look at an example:

```python
# Creating a Male class
class Male:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def info(self):
        print("Hi, I am a Male.")
        print(f"My name is {self.name}.")
        print(f"I am {self.age} years old.")
        
# Creating a Female class
class Female:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def info(self):
        print("Hi, I am a Female.")
        print(f"My name is {self.name}.")
        print(f"I am {self.age} years old.")
```

```python
# Creating instances
male1 = Male("Bijay Bose", 48)
female1 = Female("Puja Bose", 45)
```

```python
# Running a loop over the set of objects
# Calling the info() function common to both
for human in (male1, female1):
    human.info()
    print("\n")
```

Here, we have created two classes Male and Female. They share a similar structure and have the same method info(). However, notice that we have not created a common superclass or linked the classes together in any way. Even then, we can pack these two different objects into a tuple and iterate through them using a common human variable. It is possible due to polymorphism. We can call both the info() methods by just using human.info () call, where human is first male1 (Instance of Male) and then female1 (Instance of Female).


### Polymorphism and Inheritance
Like in other programming languages, the child classes in Python also inherit methods and attributes from the parent class. We can redefine certain methods and attributes specifically to fit the child class, which is known as **Method Overriding**.

Polymorphism allows us to access these overridden methods and attributes that have the same name as the parent class. Let's look at an example:

```python
# Creating a Human Class (Parent / Super class)
class Human:
    def __init__(self, name):
        self.name = name
        
    def info(self):
        print(f"Hi, I am {self.name}.")
        
# Creating a Male Class (Child / Sub class)
class Male(Human):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age
    
    def info(self):
        print(f"Hi, I am  {self.name}.")
        print(f"I am {self.age} years old.")
        print("I am a Male.")
        
# Creating a Female Class (Child / Sub class)
class Female(Human):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age
        
    def info(self):
        print(f"Hi, I am  {self.name}.")
        print(f"I am {self.age} years old.")
        print("I am a Female.")
        
```

```python
# Creating instances
human1 = Human("Aakash Dutta")
male1 = Male("Dipesh Sigha", 28)
female1 = Female("Sayani Sengupta", 25)
```

```python
# Running a loop over the set of objects
# Calling the info() function common to all the classes
for human in (human1, male1, female1):
    human.info()
    print("\n")
```

Due to polymorphism, the Python interpreter automatically recognizes that the info() method for object male1 (Male class) is overridden. So, it uses the one defined in the subclass Male. Same with the object female1 (Female Class).

**Note:** Method Overloading, a way to create multiple methods with the same name but different arguments, is not possible in Python.

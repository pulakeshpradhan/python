[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/09 Object Oriented Programming/01_Introduction_to_OOP.ipynb)

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

## Introduction to Object Oriented Programming (OOP) 
Object-Oriented Programming (OOP) is a powerful programming paradigm that allows developers to organize and structure their code in a more intuitive and modular way. It focuses on the creation of objects, which are instances of classes, and the interaction between these objects. Python, being a versatile and dynamic language, fully supports OOP principles and provides robust tools and features to implement them.


<center><img src="https://intellipaat.com/mediaFiles/2019/03/python10.png" style="max-width:550px; height: auto"></center> 


**Why do we use object-oriented programming?**
1. Modularity and code reusability.
2. Abstraction and encapsulation.
3. Organizing and maintaining code.
4. Inheritance for code reuse.
5. Flexibility through polymorphism.
6. Collaboration in team development.
7. Scalability and extensibility.


### What is an Object?
The object is an entity that has a state and behavior associated with it. It may be any real-world object like the mouse, keyboard, chair, table, pen, etc.

Integers, strings, floating-point numbers, even arrays, and dictionaries, are all objects. More specifically, any single integer or any single string is an object. The number 12 is an object, the string"  Hello, world" is an object, a list is an object that can hold other objects, and so on. You've been using objects all along and may not even realize it.


### What is a Class?
A class is a blueprint (a plan basically) that isused to define (bind together) a set of variables and methods (Characteristics) that are common to all objects of a particular kind. 

Example: If Car is a class, then Maruti 800 is an object of the Car class. All cars share similar features like wheels, 1 steeringwheel, windows, breaks, etc. Maruti 800 (the Car object) has all these features.


### Classes vs Objects (Or Instances)


* Classes are used to create user-defined data structures. Classes define functions called methods , which identify the behaviors and actions that an object created from the class can perform with its data. 

* In this module, you’ll create a Car class that stores some information about the characteristics and behaviors that an individual Car can have. 

* A class is a blueprint for how something should be defined. It doesn’t contain any data. The Car class specifies that a name and a top-speed are necessary for defining a Car , but it doesn’t contain the name or top-speed of any specific Car.

* While the class is the blueprint, an instance is an object that is built from a class and contains real data. An instance of the Car class is not a blueprint anymore. It’s an actual car with a name , like Creta, and with a top speed  of 200 Km/Hr.

* Put another way, a class is like a form or a questionnaire. An instance is like a form that has been filled out with information. Just like many people can fill out the same form with their unique information, many instances can be created from a single class.


### Defining a Class in Python

```python
# Creating a car class
class Car:
    pass
```

```python
# Creating an instance (object) from that car class
car1 = Car
# Giving some attribute to the car object
car1.name = "Maruti 800"
car1.topspeed = 120
```

```python
# Checking the attribute of the car object
print("The name of the car is", car1.name)
print("The topspeed of the car is", car1.topspeed, "Km/h")
```

```python
# Creating another car object
car2 = Car
# Giving the attribute to the new car object
car2.name = "Creta"
car2.topspeed = 400
car2.color = "Yellow"
```

```python
# Checking the attribute of the new car object
print("The name of the new car is", car2.name)
print("The topseed of the new car is", car2.topspeed, "Km/h")
print("The color of the new car is", car2.color)
```

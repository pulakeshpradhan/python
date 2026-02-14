[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Introduction_to_OOP.ipynb)

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

## Object Oriented Programming (OOP)

```python
# Creating a car class
class Car:
    pass
```

```python
# Creating an instance (Object) from car class
car1 = Car
```

```python
# Giving some attribute to the car object
car1.name = "Maruti 800"
car1.topspeed = 120
```

```python
# Print the characteristics of the car object
print("The name of the car", car1.name)
print("The topspeed of the car is", car1.topspeed, "Km/h")
```

```python
# Creating another car object
car2 = Car
```

```python
# Giving some attribute to the new car object
car2.name = "Ferrari"
car2.topspeed = 400
car2.color = "Red"
```

```python
print("The name of the car is", car2.name)
print("The topspeed of the car is", car2.topspeed, "Km/h")
print("The color of the car is", car2.color)
```

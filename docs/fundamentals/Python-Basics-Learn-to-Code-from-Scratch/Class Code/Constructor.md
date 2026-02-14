[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Constructor.ipynb)

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

## Constructor


### Defining a Class with Constructor

```python
class Car:
    # Default constructor
    def __init__(self, name, topspeed, color):
        pass
```

```python
# Creating a class with parameterized constructor
class Car:
    def __init__(self, name, topspeed, color):
        self.name = name
        self.topspeed = topspeed
        self.color = color
```

```python
# Creating an object from the car class
car1 = Car("Ferrari", 400, "Red")
car2 = Car("Maruti 800", 120, "White")
car3 = Car("Creta", 300, "Yellow")
```

```python
# Print the attributes of the car1 object
print("The name of the first car is", car1.name)
print("The topspeed of the first car is", car1.topspeed, "Km/h")
print("The color of the first car is", car1.color)
```

```python
# Print the attributes of the car2 object
print("The name of the second car is", car2.name)
print("The topspeed of the second car is", car2.topspeed, "Km/h")
print("The color of the second car is", car2.color)
```

```python
# Print the attributes of the car2 object
print("The name of the third car is", car3.name)
print("The topspeed of the third car is", car3.topspeed, "Km/h")
print("The color of the third car is", car3.color)
```

```python
# Creating the car class from scratch
class Car:
    def __init__(self, name, topspeed, color):
        self.name = name
        self.topspeed = topspeed
        self.color = color
        
    def car_info(self):
        print("Car's Information:")
        print("The name of the car is", self.name)
        print("The topspeed of the car is", self.topspeed)
        print("The color of the car is", self.color)
```

```python
# Creating 3 instances / objects from the car class
car1 = Car("Ferrari", 400, "Red")
car2 = Car("Maruti 800", 120, "White")
car3 = Car("Creta", 300, "Yellow")
```

```python
# Print the information of the car1
car1.car_info()
```

```python
# Print the information of the car2
car2.car_info()
```

```python
# Print the information of the car3
car3.car_info()
```

### Creating Application Form using Class in Python

```python
class ApplicationForm:
    def __init__(self, name, age, dob, phone):
        self.name = name
        self.age = age
        self.dob = dob
        self.phone = phone
        
    def set_email(self, email):
        self.email = email
        
    def set_address(self, address):
        self.address = address
        
    def print_form(self):
        print("Application Form:")
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"DOB: {self.dob}")
        print(f"Phone: {self.phone}")
        print(f"Emai: {self.email}")
        print(f"Address: {self.address}")
              
```

```python
# Create some object of the Application Form
form1 = ApplicationForm("Bijay Roy", 26, "2000-01-01", 6245648967)
form1.set_email("roy_bijay123@gmail.com")
form1.set_address("Newtown, Kolkata")
```

```python
# Printing the applicant form
form1.print_form()
```

```python
# Creating another object
form2 = ApplicationForm("Sovon Dey", 28, "1994-01-12", 7894256127)
form2.set_email("sovon123@gmail.com")
form2.set_address("Arambag, Hooghly")
```

```python
form2.print_form()
```

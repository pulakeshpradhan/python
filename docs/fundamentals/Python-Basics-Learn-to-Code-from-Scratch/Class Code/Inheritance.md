[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Inheritance.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="l3UMWg3uWh3k" -->
# Inheritance

<!-- #endregion -->

```python id="yRE8JWpBWciJ"
# Defining the parent/super class
class Vehicle:
  def __init__(self, brand, name, color):
    self.brand = brand
    self.name = name
    self.color = color

  def drive(self):
    print("Driving the vehicle.")

  def stop(self):
    print("Stopping the vehicle")
```

```python id="shvM7Y87Xaso"
# Defining the child/sub class
class Car(Vehicle):
  def __init__(self, brand, name, color, topspeed, no_of_windows):
    super().__init__(brand, name, color)
    self.topspeed = topspeed
    self.no_of_windows = no_of_windows

  def drive(self):
    print("Driving the car.")

  def open_trunk(self):
    print("Opening the trunk.")
```

```python id="yt1km9V5YsFo"
# Defining another child class
class Motorcycle(Vehicle):
  def __init__(self, brand, name, color, topspeed):
    super().__init__(brand, name, color)
    self.topspeed = topspeed

  def drive(self):
    print("Driving the motorcycle.")

  def indicator(self):
    print("Turning on the indicator.")
```

```python id="qtvJ3NScZkTh"
# Creating instances from subclass
car1 = Car("Maruti", "Maruti 800", "White", 120, 4)
motorcycle1 = Motorcycle("Kawasaki", "Kawasaki Ninja", "Black", 290)
```

```python colab={"base_uri": "https://localhost:8080/"} id="wVIZ6HBcaQM3" outputId="704fd665-e8f9-412e-f611-d382ea7fdff2"
# Calling the overriden methods
car1.drive()
```

```python colab={"base_uri": "https://localhost:8080/"} id="_AOOmlZVa8Wh" outputId="b7e33d5a-87fb-444b-fcad-ef8ee2eb6778"
motorcycle1.drive()
```

```python colab={"base_uri": "https://localhost:8080/"} id="MeOFx3-VbGDq" outputId="1377fa9d-ac36-43f7-b414-a929a9bb8db7"
# Calling the subclass-specific methods
car1.open_trunk()
```

```python colab={"base_uri": "https://localhost:8080/"} id="qCW_eL7PbopP" outputId="2bbcd1dc-5720-4048-967d-2d03901ef3b6"
motorcycle1.indicator()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6At2c08scjP0" outputId="a10fa5b0-213a-4042-b2c7-eebe3a4f501b"
# Calling superclass methods
car1.stop()
```

```python colab={"base_uri": "https://localhost:8080/"} id="jMCPrDj7cuye" outputId="66ebda16-384a-4ee4-ea6c-9fdbd755d6d6"
motorcycle1.stop()
```

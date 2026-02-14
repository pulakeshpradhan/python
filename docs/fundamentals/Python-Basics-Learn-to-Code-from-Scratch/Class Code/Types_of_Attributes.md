[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Types_of_Attributes.ipynb)

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

<!-- #region id="p68NJzhhNPom" -->
# Types of Attributes
<!-- #endregion -->

<!-- #region id="zzGwKXMqQ_21" -->
## Instance Attributes
<!-- #endregion -->

```python id="UyAklNp5OsFY"
class Car:
  # Creating the constructor
  # Attributes defined within the constructor are instance attributes
  def __init__(self, name, topspeed):
    self.name = name
    self.topspeed = topspeed

  def print_details(self):
    print("Car Name:", self.name)
    print("Top Speed:", self.topspeed, "Km/h")
```

```python id="SefcmFMdQBJK"
# Creating instances from Car class
car1 = Car("Maruti 800", 120)
car2 = Car("Ferrari", 400)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xqfWFYCzP-ZK" outputId="dc19a5fd-0947-4b0c-9d66-946f4eb01568"
# Printing the details of the car objects
car1.print_details()
```

```python colab={"base_uri": "https://localhost:8080/"} id="0uxlKwh3Qp4v" outputId="69480eb8-c9a9-41d8-f580-9a54c8c5add2"
car2.print_details()
```

<!-- #region id="FA2d68I_RUKw" -->
## Class Attributes
<!-- #endregion -->

```python id="uoTZ3rqQRTba"
class Car:
  # Creating a class attribute
  no_of_wheels = 4

  def __init__(self, name, topspeed):
    self.name = name
    self.topspeed = topspeed

  def print_details(self):
    print("Car Name:", self.name)
    print("Top Speed:", self.topspeed, "Km/h")
    print("No of wheels:", self.no_of_wheels)
```

```python id="R8tw1Ay2SIZC"
# Creating some instances from the new car class
car1 = Car("Creta", 300)
car2 = Car("Toyato", 240)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xGKvJT44SU46" outputId="7df8b5ca-48ce-4fa3-9e5f-53d80bfee4b5"
# Print the car details
car1.print_details()
```

```python colab={"base_uri": "https://localhost:8080/"} id="njgfCOnTSiWs" outputId="bf2f223c-6ee2-44e4-8f84-d9a108f18f85"
car2.print_details()
```

```python colab={"base_uri": "https://localhost:8080/"} id="z4bALEkQSqyV" outputId="20e83144-258b-40a9-c112-58971f65f5de"
# Calling the class attribute
car1.no_of_wheels
car2.no_of_wheels
```

```python colab={"base_uri": "https://localhost:8080/"} id="Qqf1iDcrTTWp" outputId="7b59e218-49f7-40bd-fccc-f3f80b782f70"
Car.no_of_wheels
```

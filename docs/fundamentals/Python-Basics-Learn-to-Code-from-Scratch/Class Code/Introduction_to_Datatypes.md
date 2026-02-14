[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Introduction_to_Datatypes.ipynb)

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

# Introduction to Datatypes


## Integer Datatype

```python
myInt = 10
myInt
```

```python
# To check the datatype of any variable we can use python built-in type() function
type(myInt)
```

```python
myInt2 = -10
myInt2
```

```python
type(myInt2)
```

## Float Datatype

```python
myFloat1 = 3.5
myFloat1
```

```python
type(myFloat1)
```

```python
myFloat2 = -3.5
myFloat2
```

```python
type(myFloat2)
```

## String Datatype

```python
myStr1 = "Hello World!"
myStr1
type(myStr1)
```

```python
myStr2 = "123"
type(myStr2)
```

## Boolean Datatype

```python
myBool1 = True
type(myBool1)
```

```python
myBool2 = False
type(myBool2)
```

```python
is_passed = True 
```

## List Datatype

```python
myList1 = [1, 2, 3, 4, 5]
type(myList1)
```

```python
shopping_list = ["apple", "orange", "jam", "cold-drinks", "sweets"]
shopping_list
```

```python
type(shopping_list)
```

```python
# In a list datatype, you can store different datattypes
myList2 = [1, 2.5, "apple", False]
type(myList2)
```

```python
myList3 = [1]
type(myList3)
```

## Tuple Datatype

```python
myTuple1 = (1, 2, 3, 4, 5)
type(myTuple1)
```

```python
shopping_list2 = ("orange", "mango", "milk", "brush", "ice-cream")
shopping_list2
```

```python
myTuple2 = (1, 2.5, "orange", True)
myTuple2
type(myTuple2)
```

```python
# Lists are mutable, means the value of a list item can be changed
myList1[0] = 5
myList1
```

```python
# Tuples are immutable, means the value of any tuple can not be changed.
# myTuple1[0] = 5
# myTuple1
```

```python
coords_kolkata = (22.5726, 88.3639)
```

```python
pop_kolkata  = [14900000]
```

## Set Datatype

```python
mySet1 = {1, 2, 3, 3, 2, 1}
mySet1
```

```python
mySet2 = {"Suman", "Sourin", "Bikash", "Suman"}
mySet2
```

## Dictionary Datatype

```python
myDict1 = {"fruit": "mango", "color": "yellow", "is_edible": True}
```

```python
type(myDict1)
```

```python
cityData = {
    "name": "Midnapore",
    "coords": (22.4257, 87.3199),
    "area": 18.65,
    "pin": 721101,
    "places": ["Rabindranagar", "Mohanpur"]
}
```

```python
type(cityData["area"])
```

```python
# Creating a complex dictionary
citiesData = {
    "name": "Mumbai", 
    "coords": (19.0760, 72.8777), 
    "pop": 21297000,
    "area":  603.4
}
```

```python
popDen = round(citiesData["pop"] / citiesData["area"])
print("Population Density of Mumbai City:", popDen)
```

```python
cityData2 = {
    "Kolkata": {"coords": (22.5726, 88.3639), "pop": 14900000},
    "Mumbai": {"coords": (19.0010, 72.8397), "pop": 21297000}
}
```

```python
cityData2["Kolkata"]["pop"]
```

```python
cityData2["Mumbai"]["pop"]
```

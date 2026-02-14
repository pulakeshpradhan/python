[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Break_Continue_Pass.ipynb)

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

# Break, Continue and Pass Statement

```python
cities = ["Mumbai", "Chennai", "Delhi", "Kolkata"]
```

## Break Statement

```python
print("Program Started")

for city in cities:
    if city == "Delhi":
        break
    else:
        print(city)
        
print("Program ended.")
```

```python
# First iteration, city = "Mumbai"
# Second iteration, city = "Chennai"
# Third iteration, city = "Delhi"
```

```python
i = 0
print("Program Started")

while i < 5:
    if i == 3:
        print("Executing break statement in the next statement.")
        break
    else:
        print(i)
        i += 1
```

## Continue Statement

```python
cities = ["Mumbai", "Chennai", "Delhi", "Kolkata"]
```

```python
for city in cities: # Level 1
    if city == "Delhi": # Level2 
    	continue  # Level 3
    else:  # Level 2
    	print(city) # Level 3
```

```python
i = 0
while i < 5:
    if i == 3:
        i += 1
        continue
    else:
        print(i)
        i += 1
```

## Pass Statement

```python
cities = ["Mumbai", "Chennai", "Delhi", "Kolkata"]
```

```python
for city in cities:
    pass
```

```python
i = 0
while i < 5:
    if i == 3:
        i += 1
        pass
    else:
        print(i)
        i += 1
```

```python

```

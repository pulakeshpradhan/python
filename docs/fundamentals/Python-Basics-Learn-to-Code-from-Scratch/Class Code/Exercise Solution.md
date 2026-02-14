[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/Exercise Solution.ipynb)

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

## Exercise Solution


## Temperature Converter

```python
temp = int(input("Enter the temperature in C: "))
scale = input("Enter the Scale (F, K)")
```

```python
if scale == "F":
    result = temp * (9/5) + 32
    print("The temperatur in Fahrenheit is", result)
if  scale == "K":
    result = temp + 273.15
    print("The temperatur in Kelvin is", result)
```

## Sum of N natural numbers

```python
num = int(input("Enter the number upto where you want the sum of all natural numbers: "))
```

```python
mySum = 0
for i in range(1, num+1):
    mySum += i
    
print("The sum of all natural numbers upto", num, "is")
print(mySum)
```

## Number Comparison

```python
num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))
```

```python
if num1 == num2:
    print("Both numbers are same.")
elif num1 > num2:
    print("First number is greater than the Second number.")
else:
    print("Second number is greater than the First number.")
```

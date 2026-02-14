[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/List_Methods.ipynb)

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

# List Methods

```python
myList = [1, 2, 3, 4, 5]
```

### Accessing and Modifying List Elements


### append()

```python
myList.append(6)
myList
```

### insert()

```python
myList.insert(0, 0.5)
myList
```

```python
myList.insert(4, 3.5)
myList
```

### extend()

```python
lst1 = [1, 2, 3]
lst2 = [4, 5, 6]
lst1.extend(lst2)
lst1
```

```python
lst2
```

### remove()

```python
fruitList = ["Mango", "Orange", "Watermelon", "Grapes"]
```

```python
fruitList.remove("Orange")
fruitList
```

### pop()

```python
fruitList = ["Mango", "Orange", "Watermelon", "Grapes"]
```

```python
fruitList.pop()
fruitList
```

```python
fruitList.pop(1)
fruitList
```

### index()

```python
my_nums = [1, 2, 1, 7, 5, 5, 3, 2]
```

```python
my_nums.index(7)
```

```python
my_nums.index(5)
```

```python
my_nums.index(2)
```

### count()

```python
my_nums.count(5)
```

```python
my_nums.count(7)
```

```python
my_nums.count(2)
```

### Sorting and Reversing Lists


### sort()

```python
my_list = [1, 5, 6, 4, 2, 1, 0, 9, 8]
```

```python
my_list.sort(reverse=True)
```

```python
my_list
```

### reverse()

```python
my_list
```

```python
my_list.reverse()
my_list
```

### Other List Operations


### len()

```python
len(my_list) 
```

### clear()

```python
my_list
```

```python
my_list.clear()
my_list
```

### copy()

```python
my_list1 = [1, 2, 3]
copied_list = my_list1.copy()
copied_list
```

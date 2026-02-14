[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/07 List in Python/02_List_Methods.ipynb)

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

## List Methods
In Python, a list is a versatile and commonly used data structure that allows you to store a collection of items. Lists can contain elements of different types, such as integers, strings, or even other lists. Python provides several built-in methods that allow you to perform various operations on lists efficiently.

```python
myLst1 = [1, 2, 3, 4]
myLst2 = [5, 6, 7, 8]
```

### Accessing and Modifying List Elements


### append()
The append() method allows you to add an element to the end of a list.

```python
myLst2.append(9)
myLst2
```

### insert()
The insert() method allows you to insert an element at a specific position in the list.

```python
myLst1.insert(0, 0)
myLst1
```

### extend()
The extend() method is used to append elements from another list to the end of the current list.

```python
f1 = ["Mango", "Orange"]
f2 = ["Watermelon", "Grapes"]
f1.extend(f2)
print(f1)
print(f2)
```

### remove()
The remove() method removes the first occurrence of a specified element from the list.

```python
f1
```

```python
f1.remove("Grapes")
f1
```

### pop()
The pop() method removes and returns an element at a specific index in the list. If no index is provided, it removes the last element.

```python
f1
```

```python
f1.pop(1)
f1
```

### index()
The index() method returns the index of the first occurrence of a specified element in the list.

```python
lst1 = [1, 2, 5, 7, 1, 5]
```

```python
lst1.index(5)
```

### count()
The count() method returns the number of occurrences of a specified element in the list.

```python
lst1
```

```python
lst1.count(5)
```

## Sorting and Reversing Lists


### sort()
The sort() method sorts the list in ascending order. It modifies the original list and does not return a new sorted list.

```python
lst1
```

```python
lst1.sort()
lst1
```

```python
lst2 = ["a", "c", "b"]
lst2.sort()
lst2
```

### reverse()
The reverse() method reverses the order of elements in the list. It modifies the original list and does not return a new reversed list.

```python
print(lst1)
lst1.reverse()
print(lst1)
```

## Other List Operations


### len()
The len() function returns the number of elements in a list.

```python
len(lst1)
```

### clear()
The clear() method removes all elements from a list, making it empty.

```python
c = ["a", "b", 1]
print(c)
c.clear()
print(c)
```

### copy()
The copy() method creates a shallow copy of a list. Any modifications made to the original list will not affect the copied list, and vice versa.

```python
l1 = [1, 2, 3]
id(l1)
```

```python
l2 = l1.copy()
print(l2)
print(id(l2))
```

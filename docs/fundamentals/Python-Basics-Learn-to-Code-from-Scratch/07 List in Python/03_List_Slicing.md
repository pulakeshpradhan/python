[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/07 List in Python/03_List_Slicing.ipynb)

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

## List Slicing
List slicing is a powerful feature in Python that allows you to extract a portion of a list or create a new list containing a subset of elements from an existing list. Slicing provides a concise and flexible way to work with lists and access specific elements based on their indices.


### Basic List Slicing
List slicing is performed using the colon (:) operator, which allows you to specify the start and end indices of the slice. The start index is inclusive, while the end index is exclusive, meaning that the element at the end index is not included in the slice.

```python
my_nums = [1, 2, 3, 4, 5, 6]
sliced_nums = my_nums[0:4]
sliced_nums
```

### Slicing with Step
You can also specify a step size while slicing a list. The step size determines the interval between elements in the resulting slice.

```python
my_nums
```

```python
sliced_nums_with_step = my_nums[0:5:2]
sliced_nums_with_step
```

### Negative Indexing
Python allows negative indexing, which means you can access elements from the end of the list by using negative indices. The last element has an index of -1, the second-last element has an index of -2, and so on.

```python
last_num = my_nums[-1]
last_num
```

```python
sliced_num_neg_index = my_nums[-4: -2]
sliced_num_neg_index
```

### Slicing and Assignment
List slicing can also be used to modify a slice of a list by assigning new values to the slice.

```python
my_nums
```

```python
my_nums[0:3] = [7, 8, 9]
my_nums
```

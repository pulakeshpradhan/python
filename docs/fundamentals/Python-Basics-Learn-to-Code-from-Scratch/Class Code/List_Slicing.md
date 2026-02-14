[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-Basics-Learn-to-Code-from-Scratch/Class Code/List_Slicing.ipynb)

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

```python
my_list = [1, 2, 3, 4, 5, 6] 
# I want a list of first three elements
```

```python
subset_list = my_list[0:3] #3-1
subset_list
```

```python
subset_list2 = my_list[2:5]
subset_list2
```

```python
subset_list3 = my_list[:4]
subset_list3
```

### Basic List Slicing

```python
my_nums = [1, 2, 3, 4, 5, 6]
sliced_list = my_nums[0:4]
sliced_list
```

### Slicing with Step

```python
sliced_with_step = my_nums[0:5:2]
sliced_with_step
```

### Negative Indexing

```python
my_nums = [1, 2, 3, 4, 5, 6]
```

```python
my_nums[-3]
```

```python
sliced_with_neg_index = my_nums[-3:]
sliced_with_neg_index
```

### Slicing and Assignment

```python
new_list = [1, 2, 3, 4, 5, 6]
```

```python
new_list[0:3] = [7, 8, 9]
```

```python
new_list
```

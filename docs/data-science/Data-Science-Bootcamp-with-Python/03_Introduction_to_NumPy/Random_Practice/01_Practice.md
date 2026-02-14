[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/03_Introduction_to_NumPy/Random_Practice/01_Practice.ipynb)

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

```python
import numpy as np
```

```python
myArr1D = np.array([1, 2, 3, 4])
myArr1D
```

```python
myArr2D = np.array([[1, 2, 3], [4, 5, 6]])
myArr2D
```

```python
myArr1D.ndim
```

```python
myArr2D.ndim
```

```python
myArr1D.itemsize
```

```python
myArr1D.dtype
```

```python
myArr2D = np.array([[1, 2, 3], [4, 5, 6]], dtype="float64")
myArr2D
```

```python
myArr2D.itemsize
```

```python
myArr2D.size
```

```python
myArr2D.shape
```

```python
complex_array = np.array([[1, 2], [3, 4], [5, 6]], dtype=complex)
complex_array
```

```python
np.zeros((3, 3))
```

```python
np.ones((3, 3))
```

```python
np.arange(1, 20, 2)
```

```python
np.linspace(1, 10, 5)
```

```python
np.reshape(myArr2D, (3, 2))
```

```python
myArr2D.ravel()
```

```python
myArr2D.min()
```

```python
myArr2D.max()
```

```python
myArr2D.sum()
```

```python
myArr2D
```

```python
myArr2D.sum(axis=0)
```

```python
myArr2D.sum(axis=1)
```

```python
np.sqrt(myArr1D)
```

```python
np.std(myArr2D)
```

```python
a = np.array([[1, 2], [3, 4]])
a
```

```python
b = np.array([[5, 6], [7, 8]])
b
```

```python
a + b
```

```python
a - b
```

```python
a * b
```

```python
a / b
```

```python
a.dot(b)
```

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

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/geonextgis/Data-Wrangling-with-Xarray/blob/main/00_Fundamentals/00_Data_Structures.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="sj8CioskublQ" -->
# **Xarray's Data Structures**
<!-- #endregion -->

<!-- #region id="0VmMykU3uqF_" -->
## **Introduction**
N-dimensional arrays, also known as tensors, are integral to computational science and are used across various domains like physics, astronomy, geoscience, bioinformatics, engineering, finance, and deep learning. In Python, [NumPy](https://numpy.org/) serves as the essential tool for handling these arrays. However, practical datasets go beyond simple numerical values; they often include labels that provide information about how the array values correspond to locations in space, time, and other dimensions.

To illustrate, consider how we might organize a dataset for a weather forecast:
<!-- #endregion -->

<!-- #region id="CUqcnvA6zqzA" -->
<center><img src="https://docs.xarray.dev/en/stable/_images/dataset-diagram.png" width="60%"></center>
<!-- #endregion -->

<!-- #region id="A6dQgkqh0Dlb" -->
Xarray distinguishes itself by not only keeping track of labels on arrays but using them to offer a robust and concise interface. For instance:

- Conduct operations across dimensions by name: `x.sum('time')`.
- Select values by label rather than integer location:
    `x.loc['2014-01-01']` or `x.sel(time='2014-01-01')`.
- Mathematical operations (e.g., `x - y`) efficiently work across multiple dimensions (array broadcasting) based on dimension names, not shape.
- Utilize versatile split-apply-combine operations with groupby:
    `x.groupby('time.dayofyear').mean()`.
- Achieve database-like alignment based on coordinate labels that adeptly handles missing values: `x, y = xr.align(x, y, join='outer')`.
- Retain arbitrary metadata using a Python dictionary: `x.attrs`.

Xarray's N-dimensional data structures are well-suited for handling multi-dimensional scientific data. Its use of dimension names, rather than axis labels (e.g., `dim='time'` instead of `axis=0`), makes managing arrays more straightforward compared to raw NumPy ndarrays. With xarray, there's no need to keep track of the order of an array's dimensions or insert dummy dimensions of size 1 for alignment (e.g., using `np.newaxis`).

The immediate benefit of using xarray is reduced code, and the long-term advantage is enhanced understanding when revisiting the code in the future.
<!-- #endregion -->

<!-- #region id="1-C3M4bI2ULS" -->
## **Data Structures**
Xarray offers two primary data structures: the `DataArray` and `Dataset`. The `DataArray` class adds dimension names, coordinates, and attributes to multi-dimensional arrays, while the `Dataset` class combines multiple arrays.

For practical examples, Xarray provides small real-world tutorial datasets on its GitHub repository [here](https://github.com/pydata/xarray-data). We will utilize the [xarray.tutorial.load_dataset](https://docs.xarray.dev/en/stable/generated/xarray.tutorial.open_dataset.html#xarray.tutorial.open_dataset) function to download and open the `air_temperature` Dataset from the National Centers for Environmental Prediction by name.
<!-- #endregion -->

```python id="Vo1sfSUM9g6v"
import numpy as np
import xarray as xr
```

<!-- #region id="yfeTsE9B3IfG" -->
### **Dataset**
`Dataset` objects function as container-like structures resembling dictionaries. They organize DataArrays, where each variable name is mapped to an associated DataArray within the dataset. This arrangement allows for a comprehensive and structured representation of multi-variable datasets.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 374} id="XubFsNNy9x4Q" outputId="4da71b4a-2073-4a73-ed94-526927885f37"
# Reading built-in dataset with Xarray
ds = xr.tutorial.load_dataset("air_temperature")
ds
```

<!-- #region id="UzxLvL1U-FZz" -->
We can access "layers" of the Dataset (individual DataArrays) with dictionary syntax.
<!-- #endregion -->

```python id="F9fhaQ7S_ADh"
ds["air"]
```

<!-- #region id="_npCn9ZO_JjT" -->
We can save some typing by using the "attribute" or "dot" notation. This won't work for variable names that clash with built-in method names (for example, `mean`).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 219} id="T5A_6o_mDeoT" outputId="5c5f7f44-9a0c-4af0-9d72-7e6b42407f37"
ds.air
```

<!-- #region id="Kxbz_AuoDi_U" -->
#### **Understanding String Representations**

Xarray offers two types of representations: `"html"` (exclusive to notebooks) and `"text"`. You can specify your preference using the `display_style` option.

Up to this point, our notebook has been set to automatically display the `"html"` representation (which we will stick with). The `"html"` representation is interactive, enabling you to collapse sections (using left arrows) and explore attributes and values for each entry (accessible through the right-hand sheet icon and data symbol).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 374} id="2DKnYgOFGX0v" outputId="a34a3edb-558f-4768-df8a-547da0e77ecc"
with xr.set_options(display_style="html"):
    display(ds)
```

<!-- #region id="uKUz1ynSHVL8" -->
The output includes:

- A summary detailing all *dimensions* of the `Dataset` `(lat: 25, time: 2920, lon: 53)`. This information specifies that the first dimension, named `lat`, has a size of `25`, the second dimension, named `time`, has a size of `2920`, and the third dimension, named `lon`, has a size of `53`. Since we access dimensions by name, their order is not significant.
- An unordered list presenting *coordinates* or dimensions with coordinates. Each item is listed on a separate line, providing the name, one or more dimensions in parentheses, the data type (dtype), and a preview of the values. Additionally, if a dimension coordinate is present, it is marked with a `*`.
- An alphabetically sorted list of *dimensions without coordinates* (if any).
- An unordered list detailing *attributes*, or metadata.
<!-- #endregion -->

<!-- #region id="kP27qqBXHEDD" -->
🤔 **Note:** The use of the `with` statement in Python is associated with context management. In this context, the `xr.set_options(display_style="html")` is likely a context manager provided by the xarray library. When used within a `with` statement, it allows you to temporarily change a setting for a specific block of code, and once the block is exited, the original settings are automatically restored.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 260} id="DUGp_lfoICPI" outputId="674d68a9-2f78-4dba-a091-322e40768889"
with xr.set_options(display_style="text"):
    display(ds)
```

<!-- #region id="kL6_nXYtIP3b" -->
To understand each of the components better, we'll explore the "air" variable of our Dataset.
<!-- #endregion -->

<!-- #region id="FYO3Iuc4IiP1" -->
### **DataArray**
The `DataArray` class consists of an array (data) and its associated dimension names, labels, and attributes (metadata).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 219} id="6b1wojjFI94P" outputId="e74e4b4c-1945-4b8f-be22-231933e9f9d6"
# Selecting 'air' variable from the dataset
da = ds["air"]
da
```

<!-- #region id="1bY-dYooV1zT" -->
#### **Understanding String Representations**
We can use the same two representations (`"html"`, which is only available in
notebooks, and `"text"`) to display our `DataArray`.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 219} id="iJ7BHrS1V7cz" outputId="41640828-354b-4124-aa58-61c7de573c0b"
with xr.set_options(display_style="html"):
    display(da)
```

```python id="RA4F0D9HhkaG" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="f92d89f2-e48d-42d4-cc84-245c785c3992"
with xr.set_options(display_style="text"):
    display(da)
```

<!-- #region id="CNsXMte6n_yx" -->
We can also access the data array directly:
<!-- #endregion -->

```python id="wZyz24vioBeS"
ds.air.data # (or equivalently, `da.data`)
```

<!-- #region id="T68r_uRkoZUd" -->
#### **Named Dimensions**
`.dims` represent the named axes of your data, and they can either have associated values (dimension coordinates) or not (dimensions without coordinates). The names can take any form that is compatible with a Python `set` (i.e., calling `hash()` on it does not result in an error), but for practical use, they are typically strings.

In this instance, there are two spatial dimensions, with shorthand names `lat` and `lon` representing `latitude` and `longitude`, respectively. Additionally, there is one temporal dimension, denoted as `time`.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="l-9bvHEFrJSG" outputId="7ce9cd63-38d9-44f7-f228-e3814d21f36d"
ds.air.dims
```

<!-- #region id="EgXoySBhrPo4" -->
#### Coordinates

`.coords` serves as a straightforward [dict-like](https://docs.python.org/3/glossary.html#term-mapping) [data container](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates) that maps coordinate names to corresponding values. These values can take different forms:

- Another `DataArray` object.
- A tuple `(dims, data, attrs)`, where `attrs` is optional. This is akin to creating a new `DataArray` object with `DataArray(dims=dims, data=data, attrs=attrs)`.
- A 1-dimensional `numpy` array or any convertible type (using [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html)), such as a `list`. This array contains numbers, datetime objects, strings, etc., serving as labels for each point.

In the following example, we observe the actual timestamps and spatial positions associated with our air temperature data:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="OxvJIcyBrrdC" outputId="e194de16-0ff3-4d87-d021-99225986ae90"
ds.air.coords
```

<!-- #region id="7wv7ZOhTr-D7" -->
The distinction between dimension labels (dimension coordinates) and regular coordinates lies in the fact that, currently, indexing operations (`sel`, `reindex`, etc.) can only be applied to dimension coordinates. Additionally, while coordinates can have arbitrary dimensions, it is a requirement for dimension coordinates to be one-dimensional.
<!-- #endregion -->

<!-- #region id="AP1TTCz7sSU8" -->
### **Attributes**
`.attrs` is a dictionary capable of holding diverse Python objects, including strings, lists, integers, dictionaries, etc., to store information about your data. The only constraint is that certain attributes might not be writable to specific file formats.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="zEt5aY7Xseew" outputId="fc5f4ea2-970e-4e64-b8e1-7c41dbcf44f3"
ds.air.attrs
```

<!-- #region id="vqGO7UgLvhY7" -->
## **Bridging Pandas and Xarray**
Frequently, the creation of `DataArray` and `Dataset` objects involves conversions from other libraries like [pandas](https://pandas.pydata.org/) or by reading data from storage formats such as [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) or [zarr](https://zarr.readthedocs.io/en/stable/).

To facilitate conversion between `xarray` and `pandas`, you can utilize the [to_xarray](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_xarray.html) methods on Pandas objects or the [to_pandas](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.to_pandas.html) methods on `xarray` objects:
<!-- #endregion -->

```python id="x75fOW1pvtZM"
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/"} id="F9t69wSZwaAP" outputId="4745a914-0967-472e-b1c7-078b91df5bcf"
series = pd.Series(np.ones((10,)), index=list("abcdefghij"))
series
```

```python colab={"base_uri": "https://localhost:8080/", "height": 177} id="CP7tbyxlw5E0" outputId="fd5078a1-30bc-440c-c5a0-384bad40ca2e"
arr = series.to_xarray()
arr
```

```python colab={"base_uri": "https://localhost:8080/"} id="Bu5_DV6rxOcL" outputId="cdaa816c-a8f2-45b3-c370-b2157293d6eb"
arr.to_pandas()
```

<!-- #region id="aBZ_90xpxUD0" -->
We can also control what `pandas` object is used by calling `to_series` /
`to_dataframe`:

<!-- #endregion -->

<!-- #region id="Bbketlp6x0WJ" -->
**`to_series`**: This will always convert `DataArray` objects to
`pandas.Series`, using a `MultiIndex` for higher dimensions

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="E4wNSCuuxWp1" outputId="a35ddc17-cac0-4537-979f-80be6d334520"
ds.air.to_series()
```

<!-- #region id="UkXzx3iTyFRK" -->
**`to_dataframe`**: This will always convert `DataArray` or `Dataset`
objects to a `pandas.DataFrame`. Note that `DataArray` objects have to be named
for this.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 455} id="eiVzy9Q-xdCg" outputId="eb493513-9ac5-4ef8-d500-f8640fb51156"
ds.air.to_dataframe()
```

<!-- #region id="3-uMQRk5ya6t" -->
Since columns in a `DataFrame` need to have the same index, they are
broadcasted.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 455} id="tnH4ibc9xqjQ" outputId="208d1572-e1cd-42fd-829f-250f98dc057d"
ds.to_dataframe()
```

<!-- #region id="fB8XzuVcVi8D" -->
## **To Pandas and back**
`DataArray` and `Dataset` objects are commonly generated through the conversion of data from other libraries like pandas or by reading from various data storage formats such as NetCDF or zarr.

To convert from / to `pandas`, we can use the `to_xarray` methods on pandas objects or the `to_pandas` methods on `xarray` objects:
<!-- #endregion -->

```python id="lNRIRHrQVl9F"
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/"} id="0yGLMXPJXCdH" outputId="cd44c701-37cc-4154-8cb6-aaa8c4868209"
series = pd.Series(np.ones((10,)), index=list("abcdefghij"))
series
```

```python colab={"base_uri": "https://localhost:8080/", "height": 177} id="xxrSEdoiXXaR" outputId="ad558a15-79d2-456b-e011-54058489e99c"
arr = series.to_xarray()
arr
```

```python colab={"base_uri": "https://localhost:8080/"} id="hhYiCkjVXQLn" outputId="adc58d5c-5d62-417c-a11e-a4967619154d"
arr.to_pandas()
```

<!-- #region id="nmNUNZP-XpGK" -->
We can also control what `pandas` object is used by calling `to_series` /
`to_dataframe`:

**`to_series`**: This will always convert `DataArray` objects to
`pandas.Series`, using a `MultiIndex` for higher dimensions

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="UAHtKZ8eYEAW" outputId="b46cca99-9fdf-4348-b8dc-a97230e086a2"
ds.air.to_series()
```

<!-- #region id="KwVsMEqoYIrO" -->
**`to_dataframe`**: This will always convert `DataArray` or `Dataset`
objects to a `pandas.DataFrame`. Note that `DataArray` objects have to be named
for this.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 455} id="hsqwOIUZYNi1" outputId="69b4cf47-b592-470a-ed5a-87635bc6e6b9"
ds.air.to_dataframe()
```

<!-- #region id="4yvLZh36YhRX" -->
Since columns in a `DataFrame` need to have the same index, they are
broadcasted.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 455} id="J_hbaW0ZYj8g" outputId="677b89fa-79b0-4d9c-c65a-85ec3d5d848e"
ds.to_dataframe()
```

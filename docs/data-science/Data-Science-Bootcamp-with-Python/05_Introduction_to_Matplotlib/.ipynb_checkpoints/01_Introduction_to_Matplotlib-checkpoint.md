# Introduction to Matplotlib
Matplotlib is a widely used Python library for data visualization. It provides a variety of functions for creating different types of graphs and charts. Matplotlib can be used to create simple line plots, scatter plots, histograms, bar charts, 3D plots, and more. In this module, we will cover the basics of Matplotlib and how to use it for data visualization.

**Prerequisites:**
* Basic understanding of Python
* Familiarity with NumPy library is recommended but not required.

<center><img src="https://matplotlib.org/_static/logo_dark.svg" style= "max-width: 350px; height: auto"></center>

## 01. Installation and Importing
To install Matplotlib, use the pip package manager in the terminal by typing the following command:


```python
# !pip install matplotlib
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
```

## 02. Plotting Simple Data
Let's start by creating a simple line plot.


```python
# Creating two variable
x = [1, 2, 3, 4, 5, 6, 7]
y = [40, 38, 43, 45, 42, 40, 39]
```


```python
plt.plot(x, y)
```

## 03. API for Plot
In Matplotlib, you can customize the appearance of lines in a plot using the color, linewidth, and linestyle arguments in the plot() function.

* The color argument sets the color of the line. It can be specified using a string such as "red" or "blue", a hex code such as "#FF5733", or an RGB tuple such as (0.5, 0.5, 0.5).

* The linewidth argument sets the width of the line. It can be specified using a floating-point number.

* The linestyle argument sets the style of the line. It can be specified using a string such as "solid", "dashed", or "dotted", or a combination of those using a ":" for dots and a "--" for dashes.


```python
# Creating a new plot
plt.plot(x, y, color="red", linewidth=4, linestyle="dotted")
```

## 04. Adding Title and Labels to the Chart
In Matplotlib, you can add titles and labels to your plot to provide context and improve its readability.

* To add a title to your plot, you can use the title() method of the axes object. This method takes a string argument that specifies the title of the plot.

* To add labels to the x-axis and y-axis, you can use the xlabel() and ylabel() methods of the axes object, respectively. These methods take a string argument that specifies the label of the axis. 


```python
# Creating another plot
plt.plot(x, y, color="green", linewidth=4, linestyle="dashed")
plt.title("Weather")
plt.xlabel("Day")
plt.ylabel("Temperature")
```

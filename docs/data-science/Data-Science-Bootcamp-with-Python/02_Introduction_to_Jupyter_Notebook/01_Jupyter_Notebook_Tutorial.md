[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/Data-Science-Bootcamp-with-Python/02_Introduction_to_Jupyter_Notebook/01_Jupyter_Notebook_Tutorial.ipynb)

# Jupyter Notebook Tutorial

## 01. Working with Codes


```python
# Import the pandas library and give it an alias 'pd'
import pandas as pd

# Define the URL where the Iris dataset is hosted as a csv
url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"

# Use pandas to read the CSV file located at the specified URL into a DataFrame
df = pd.read_csv(url)

# Print the first 5 rows of the DataFrame to verify that the data was loaded correctly
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal.length</th>
      <th>sepal.width</th>
      <th>petal.length</th>
      <th>petal.width</th>
      <th>variety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
  </tbody>
</table>
</div>



## 02. Working with Plots


```python
%matplotlib inline
df.plot(y="petal.length", color="green")
```




    <Axes: >




    
![png](01_Jupyter_Notebook_Tutorial_files/01_Jupyter_Notebook_Tutorial_4_1.png)
    



```python
df.plot.bar(x="variety", y="petal.length")
```




    <Axes: xlabel='variety'>




    
![png](01_Jupyter_Notebook_Tutorial_files/01_Jupyter_Notebook_Tutorial_5_1.png)
    


## 03. Working with Magic Commands


```python
%lsmagic
```




    Available line magics:
    %alias  %alias_magic  %autoawait  %autocall  %automagic  %autosave  %bookmark  %cd  %clear  %cls  %colors  %conda  %config  %connect_info  %copy  %ddir  %debug  %dhist  %dirs  %doctest_mode  %echo  %ed  %edit  %env  %gui  %hist  %history  %killbgscripts  %ldir  %less  %load  %load_ext  %loadpy  %logoff  %logon  %logstart  %logstate  %logstop  %ls  %lsmagic  %macro  %magic  %matplotlib  %mkdir  %more  %notebook  %page  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %pip  %popd  %pprint  %precision  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab  %qtconsole  %quickref  %recall  %rehashx  %reload_ext  %ren  %rep  %rerun  %reset  %reset_selective  %rmdir  %run  %save  %sc  %set_env  %store  %sx  %system  %tb  %time  %timeit  %unalias  %unload_ext  %who  %who_ls  %whos  %xdel  %xmode
    
    Available cell magics:
    %%!  %%HTML  %%SVG  %%bash  %%capture  %%cmd  %%debug  %%file  %%html  %%javascript  %%js  %%latex  %%markdown  %%perl  %%prun  %%pypy  %%python  %%python2  %%python3  %%ruby  %%script  %%sh  %%svg  %%sx  %%system  %%time  %%timeit  %%writefile
    
    Automagic is ON, % prefix IS NOT needed for line magics.




```python
%time for i in range(1, 10000): i*i
```

    CPU times: total: 0 ns
    Wall time: 1.05 ms
    

## 04. Adding YouTube Videos


```python
%%HTML
<iframe width="560" height="315" src="https://www.youtube.com/embed/6_jEgQjwfok" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
```


<iframe width="560" height="315" src="https://www.youtube.com/embed/6_jEgQjwfok" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>



## 05. Adding Markdown

This is an example of a Markdown document. Markdown is a lightweight markup language that is used to format plain text documents. It's easy to learn and use, and it's supported by many tools and platforms.

### Headers
You can use hash symbols (`#`) to create headers of different levels. For example:
# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6

### Text Formatting
You can use asterisks (`*`) or underscores (`_`) to add emphasis to text. For example:

*This text will be italicized.*
_This text will also be italicized._

**This text will be bold.**
__This text will also be bold.__

You can combine these characters to create different effects, like so:

**_This text will be bold and italicized._**

### Lists
You can create ordered and unordered lists using hyphens (`-`) or numbers (`1.`). For example:

#### Unordered List
- Item 1
- Item 2
- Item 3

#### Ordered List
1. First item
2. Second item
3. Third item

### Links
You can create links to other websites or pages using square brackets (`[]`) and parentheses (`()`). For example:

[Click here to visit GeoNext](https://dev-geonext.pantheonsite.io/)

### Images
You can include images in your Markdown document using the same syntax as links, but with an exclamation mark (`!`) at the beginning. For example:

![GeoNext Logo](https://dev-geonext.pantheonsite.io/wp-content/uploads/2022/09/GeoNext-Logo.png)
<center><img src="https://dev-geonext.pantheonsite.io/wp-content/uploads/2022/09/GeoNext-Logo.png" style="max-width: 250px; height: auto;"></center>

### Code
You can include code snippets in your Markdown document by wrapping the code in backticks (` ` `) or by using a code block. For example:

```python
print("Hello World!")
```



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Exercises/Python Exercise (91-100).ipynb)

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
    language: python
    name: python3
---

# Python Exercise (91-100)


# Question 91
>***Please write a program which accepts a string from console and print it in reverse order.***

>**Example:
If the following string is given as input to the program:***
```
rise to vote sir
```
>***Then, the output of the program should be:***
```
ris etov ot esir
```
### Hints 
> ***Use list[::-1] to iterate a list in a reverse order.***

```python

```

## Question 92

>***Please write a program which accepts a string from console and print the characters that have even indexes.***

>***Example:
If the following string is given as input to the program:***
```
H1e2l3l4o5w6o7r8l9d
```
>***Then, the output of the program should be:***
```
Helloworld
```
### Hints 
>***Use list[::2] to iterate a list by step 2.***


```python

```

## Question 93

>***Please write a program which prints all permutations of [1,2,3]***

----------------------
### Hints 
> ***Use itertools.permutations() to get permutations of list.***

```python

```

## Question 94

>***Write a program to solve a classic ancient Chinese puzzle: 
We count 35 heads and 94 legs among the chickens and rabbits in a farm. How many rabbits and how many chickens do we have?***

```python

```

## Question 95
>***Given the participants' score sheet for your University Sports Day, you are required to find the runner-up score. You are given  scores. Store them in a list and find the score of the runner-up.***

>***If the following string is given as input to the program:***
>```
>5
>2 3 6 6 5
>```
>***Then, the output of the program should be:***
>```
>5
>```
### Hints 
> ***Make the scores unique and then find 2nd best number***


```python

```

## Question 96
>***You are given a string S and width W.
Your task is to wrap the string into a paragraph of width.***

>***If the following string is given as input to the program:***
>```
>ABCDEFGHIJKLIMNOQRSTUVWXYZ
>4
>```
>***Then, the output of the program should be:***
>```
>ABCD
>EFGH
>IJKL
>IMNO
>QRST
>UVWX
>YZ
>```

### Hints
> ***Use wrap function of textwrap module***|

```python

```

<!-- #region -->
## Question 97


>***You are given an integer, N. Your task is to print an alphabet rangoli of size N. (Rangoli is a form of Indian folk art based on creation of patterns.)***

>***Different sizes of alphabet rangoli are shown below:***
>```
>#size 3
>
>----c----
>--c-b-c--
>c-b-a-b-c
>--c-b-c--
>----c----
>
>#size 5
>
>--------e--------
>------e-d-e------
>----e-d-c-d-e----
>--e-d-c-b-c-d-e--
>e-d-c-b-a-b-c-d-e
>--e-d-c-b-c-d-e--
>----e-d-c-d-e----
>------e-d-e------
>--------e--------
>```
### Hints 
>***First print the half of the Rangoli in the given way and save each line in a list. Then print the list in reverse order to get the rest.***
<!-- #endregion -->

```python

```

<!-- #region -->
## Question 98

>***You are given a date. Your task is to find what the day is on that date.***

**Input**
>***A single line of input containing the space separated month, day and year, respectively, in MM DD YYYY format.***
>```
>08 05 2015
>```


**Output**
>***Output the correct day in capital letters.***
>```
>WEDNESDAY
>```


----------------------
### Hints 
> ***Use weekday function of calender module***
<!-- #endregion -->

```python

```

<!-- #region -->
## Question 99
>***Given 2 sets of integers, M and N, print their symmetric difference in ascending order. The term symmetric difference indicates those values that exist in either M or N but do not exist in both.***

**Input**
>***The first line of input contains an integer, M.The second line contains M space-separated integers.The third line contains an integer, N.The fourth line contains N space-separated integers.***
>```
>4
>2 4 5 9
>4
>2 4 11 12
>```

**Output**
>***Output the symmetric difference integers in ascending order, one per line.***
>```
>5
>9
>11
>12
>```


----------------------
### Hints
> ***Use \'^\' to make symmetric difference operation.***


<!-- #endregion -->

```python

```

## Question 100

>***You are given  words. Some words may repeat. For each word, output its number of occurrences. The output order should correspond with the input order of appearance of the word. See the sample input/output for clarification.***

>***If the following string is given as input to the program:***
>```
>4
>bcdef
>abcdefg
>bcde
>bcdef
>```
>***Then, the output of the program should be:***
>```
>3
>2 1 1
>```

### Hints 
> ***Make a list to get the input order and a dictionary to count the word frequency***



```python

```

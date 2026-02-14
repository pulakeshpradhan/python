[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Exercises/Python_Exercise_(1_10).ipynb)

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

<!-- #region id="iNNiPxIo_XEi" -->
# Python Exercise (1-10)
<!-- #endregion -->

<!-- #region id="kcCG8xdk_XEl" -->
## Question 1


Write a program which will find all such numbers which are divisible by 7 but are not a multiple of 5,
between 2000 and 3200 (both included).
The numbers obtained should be printed in a comma-separated sequence on a single line.


<!-- #endregion -->

```python vscode={"languageId": "plaintext"}

```

<!-- #region id="Yig61r6o_XEn" -->
## Question 2

Write a program which can compute the factorial of a given numbers.
The results should be printed in a comma-separated sequence on a single line.
Suppose the following input is supplied to the program:
8
Then, the output should be:
40320

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
<!-- #endregion -->

```python vscode={"languageId": "plaintext"}

```

<!-- #region id="QcYcdIEl_XEp" -->
## Question 3

With a given integral number n, write a program to generate a dictionary that contains (i, i*i) such that is an integral number between 1 and n (both included). and then the program should print the dictionary.
Suppose the following input is supplied to the program:
8
Then, the output should be:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
Consider use dict()
<!-- #endregion -->

```python vscode={"languageId": "plaintext"}

```

<!-- #region id="QzqHCtDp_XEp" -->
## Question 4

Write a program which accepts a sequence of comma-separated numbers from console and generate a list and a tuple which contains every number.
Suppose the following input is supplied to the program:
34,67,55,33,12,98
Then, the output should be:
['34', '67', '55', '33', '12', '98']
('34', '67', '55', '33', '12', '98')

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
tuple() method can convert list to tuple
<!-- #endregion -->

```python vscode={"languageId": "plaintext"}

```

<!-- #region id="nbAvUxZy_XEq" -->
## Question 5


Define a class which has at least two methods:
getString: to get a string from console input
printString: to print the string in upper case.
Also please include simple test function to test the class methods.

Hints:
Use __init__ method to construct some parameters
<!-- #endregion -->

```python vscode={"languageId": "plaintext"}

```

<!-- #region id="KsweINJ__XEq" -->
## Question 6


Write a program that calculates and prints the value according to the given formula:
Q = Square root of [(2 * C * D)/H]
Following are the fixed values of C and H:
C is 50. H is 30.
D is the variable whose values should be input to your program in a comma-separated sequence.
Example
Let us assume the following comma separated input sequence is given to the program:
100,150,180
The output of the program should be:
18,22,24

Hints:
If the output received is in decimal form, it should be rounded off to its nearest value (for example, if the output received is 26.0, it should be printed as 26)
In case of input data being supplied to the question, it should be assumed to be a console input.
<!-- #endregion -->

```python vscode={"languageId": "plaintext"}

```

<!-- #region id="BWiP0z39_XEq" -->
## Question 7

Write a program which takes 2 digits, X,Y as input and generates a 2-dimensional array. The element value in the i-th row and j-th column of the array should be i*j.
Note: i=0,1.., X-1; j=0,1,¡­Y-1.
Example
Suppose the following inputs are given to the program:
3,5
Then, the output of the program should be:
[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]

Hints:
Note: In case of input data being supplied to the question, it should be assumed to be a console input in a comma-separated form.
<!-- #endregion -->

```python vscode={"languageId": "plaintext"}

```

<!-- #region id="wZjooJZf_XEr" -->
## Question 8

Write a program that accepts a comma separated sequence of words as input and prints the words in a comma-separated sequence after sorting them alphabetically.Suppose the following input is supplied to the program:
without,hello,bag,world

Then, the output should be:
bag,hello,without,world
<!-- #endregion -->

```python vscode={"languageId": "plaintext"}

```

<!-- #region id="nsGKryzP_XEr" -->
## Question 9

Write a program that accepts sequence of lines as input and prints the lines after making all characters in the sentence capitalized.
Suppose the following input is supplied to the program:

```
Hello world
Practice makes perfect
```
Then, the output should be:
```
HELLO WORLD
PRACTICE MAKES PERFECT
```
Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
<!-- #endregion -->

```python vscode={"languageId": "plaintext"}

```

<!-- #region id="jwuSEoYo_XEr" -->
## Question 10

Write a program that accepts a sequence of whitespace separated words as input and prints the words after removing all duplicate words and sorting them alphanumerically.
Suppose the following input is supplied to the program:
```
hello world and practice makes perfect and hello world again
```
Then, the output should be:
```
again and hello makes perfect practice world
```
Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
We use set container to remove duplicated data automatically and then use sorted() to sort the data.
<!-- #endregion -->

```python vscode={"languageId": "plaintext"}

```

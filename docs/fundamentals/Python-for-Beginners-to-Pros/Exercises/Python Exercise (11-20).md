[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Exercises/Python Exercise (11-20).ipynb)

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

# Python Exercise (11-20)


## Question 11

Write a program which accepts a sequence of comma separated 4 digit binary numbers as its input and then check whether they are divisible by 5 or not. The numbers that are divisible by 5 are to be printed in a comma separated sequence.

Example:
```
0100,0011,1010,1001
```
Then the output should be:
```
1010
```



```python

```

## Question 12

Write a program, which will find all such numbers between 1000 and 3000 (both included) such that each digit of the number is an even number.
The numbers obtained should be printed in a comma-separated sequence on a single line.

```python

```

## Question 13

Write a program that accepts a sentence and calculate the number of letters and digits.
Suppose the following input is supplied to the program:
```
hello world! 123
```
Then, the output should be:
```
LETTERS 10
DIGITS 3
```

```python

```

## Question 14

Write a program that accepts a sentence and calculate the number of upper case letters and lower case letters.
Suppose the following input is supplied to the program:
```
Hello world!
```
Then, the output should be:
```
UPPER CASE 1
LOWER CASE 9
```

```python

```

## Question 15

Write a program that computes the value of a+aa+aaa+aaaa with a given digit as the value of a.
Suppose the following input is supplied to the program:
```
9
```
Then, the output should be:
```
11106
```


```python

```

## Question 16

Use a list comprehension to select each odd number in a list. The list is input by a sequence of comma-separated numbers.
Suppose the following input is supplied to the program:
```
1,2,3,4,5,6,7,8,9
```
Then, the output should be:
```
1,3,5,7,9
```


```python

```

## Question 17

Question:
Write a program that computes the net amount of a bank account based a transaction log from console input. The transaction log format is shown as following:
```
D 100
W 200
```
D means deposit while W means withdrawal.
Suppose the following input is supplied to the program:
```
D 300
D 300
W 200
D 100

```
Then, the output should be:
```
500
```

```python

```

## Question 18
A website requires the users to input username and password to register. Write a program to check the validity of password input by users.
Following are the criteria for checking the password:
1. At least 1 letter between [a-z]
2. At least 1 number between [0-9]
1. At least 1 letter between [A-Z]
3. At least 1 character from [$#@]
4. Minimum length of transaction password: 6
5. Maximum length of transaction password: 12

Your program should accept a sequence of comma separated passwords and will check them according to the above criteria. Passwords that match the criteria are to be printed, each separated by a comma.

If the following passwords are given as input to the program:

```
ABd1234@1,a F1#,2w3E*,2We3345
```
Then, the output of the program should be:
```
ABd1234@1

```

```python

```

## Question 19

You are required to write a program to sort the (name, age, height) tuples by ascending order where name is string, age and height are numbers. The tuples are input by console. The sort criteria is:
1: Sort based on name;
2: Then sort based on age;
3: Then sort by score.
The priority is that name > age > score.
If the following tuples are given as input to the program:

```
Tom,19,80
John,20,90
Jony,17,91
Jony,17,93
Json,21,85
```
Then, the output should be:
```
[('John', '20', '90'), ('Jony', '17', '91'), ('Jony', '17', '93'), ('Json', '21', '85'), ('Tom', '19', '80')]
```


```python

```

## Question 20

Define a class with a generator which can iterate the numbers, which are divisible by 7, between a given range 0 and n.

```python

```

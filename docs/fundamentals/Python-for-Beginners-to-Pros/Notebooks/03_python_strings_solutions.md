[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Notebooks/03_python_strings_solutions.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: py310
    language: python
    name: python3
---

# **Python Strings - Exercises**

```python
# Find the length of a given string without using the len() function

len_of_string = 0

str_input = input("Enter your string: ")

for i in str_input:
    len_of_string += 1

print("Input string:", str_input)
print("The length of the string is:", len_of_string)
```

```python
# Extract username from a given email. 
# Eg if the email is halder23krishnagopal@gmail.com 
# then the username should be halder23krishnagopal

email = input("Enter your email: ")

username = email.split("@")[0]

print("Email: ", email)
print("Username:", username)
```

```python
# Count the frequency of a particular character in a provided string. 
# Eg 'hello how are you' is the string, the frequency of h in this string is 2.

input_string = input("Enter your string: ")
input_character = input("Enter the character: ")

char_frequency = input_string.count(input_character)

print("Input string: ", input_string)
print("The frequency of", input_character, "in the string is", char_frequency)
```

```python
# Write a program which can remove a particular character from a string.

input_string = input("Enter your string: ")
char_to_be_removed = input("Enter the character to be removed: ")
new_string = ""

for i in input_string:
    if i == char_to_be_removed:
        continue
    new_string = new_string + i

print("Input string: ", input_string)
print("Character to be removed: ", char_to_be_removed)
print("New string:", new_string)
```

```python
# Write a program that can check whether a given string is palindrome or not.
# abba
# malayalam

input_string = input("Enter your string: ")

reversed_string = input_string[::-1]

print("Input string: ", input_string)
print("Reversed string: ", reversed_string)

if input_string == reversed_string:
    print("The string is palindrome.")

else:
    print("The string is not a palindrome.")
```

```python
# Write a program to count the number of words in a string without split()

input_string = input("Enter your string: ")

words = list()
number_of_words = 0
word = ""

for i in input_string:
    if i != " ":
        word = word + i

    else:
        words.append(word)
        word = ""
words.append(word)

print("Input string: ", input_string)
print("Words:", words)
print("The number of words in the string:", len(words))
```

```python
# Write a python program to convert a string to title case without using the title()

input_string = input("Enter your string: ")
title_words = []

for word in input_string.split(" "):
    new_word = word[0].upper() + word[1:].lower()
    title_words.append(new_word)

print("Input string: ", input_string)
print("String in title case: ", " ".join(title_words))
```

```python
# Write a program that can convert an integer to string.

input_int = input("Enter your integer: ")

print("Input integer:", int(input_int))
print("Datatype:", type(input_int))
```

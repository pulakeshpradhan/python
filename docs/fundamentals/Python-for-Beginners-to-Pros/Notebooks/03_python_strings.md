[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Notebooks/03_python_strings.ipynb)

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

<!-- #region tags=["parameters"] -->
# **Python Strings**
In Python, a string is a sequence of characters used to represent text. Strings are enclosed in quotes, either single (`'`), double (`"`), triple single (`'''`), or triple double (`"""`). They are immutable, meaning they cannot be changed after creation. Strings can be concatenated, sliced, and have various methods for manipulation like `.lower()`, `.upper()`, and `.split()`. They also support formatting through f-strings and the `format()` method.
<!-- #endregion -->

## **Creating String**

```python
# Single-line strings
myStr = 'Hello'
myStr = "Hello"

# Multi-line strings
myStr = '''Hello'''
myStr = """Hello"""

# Typecasting
myStr = str(myStr)

print(myStr)
```

The format `"It's raining outside."` demonstrates the use of single quotes inside a string enclosed in double quotes. This is useful for including an apostrophe in the text without needing an escape character.

```python
"It's raining outside."
```

## **Accessing Substrings from a String**

In Python, both positive and negative indexing are used to access elements in sequences such as strings, lists, and tuples.

- **Positive Indexing:** Positive indexing starts from 0, with the first element of the sequence having an index of 0, the second element an index of 1, and so on.

- **Negative Indexing:** Negative indexing starts from -1, with the last element of the sequence having an index of -1, the second-to-last element an index of -2, and so on.

```python
# Positive Indexing
myStr = "hello world"
print(myStr[0])
print(myStr[6])
# print(myStr[42]) # outputs error
```

```python
# Negative Indexing
print(myStr[-1])
print(myStr[-5])
# print(myStr[-67]) # outputs error
```

```python
# Slicing
strSlice = myStr[:5]
print(strSlice)
print(myStr[1:5])
print(myStr[:])
print(myStr[0:6:2]) # With steps
print(myStr[::-1]) # Reverse string
```


When slicing with negative indexing in Python, the start index should be greater than the stop index for the slice to include elements from left to right in the sequence. This is because negative indexing counts from the end of the sequence.

```python
print(myStr[-5:])
print(myStr[-1:-6:-1])
```

## **Editing and Deleting Strings**
Strings in Python are immutable, which means once they are created, they cannot be changed. However, you can create new strings based on modifications to existing ones. Additionally, while you cannot delete a part of a string, you can delete the entire string.

```python
myStr = "hello world"
# myStr[0] = "H" # outputs error
```

```python
# Delete a string
del myStr
```

```python
# # Quick question
# # Guess the output of this code cell
# myStr = "hello world"
# del myStr[-1:-5:-2]
# print(myStr)
```

## **Operations on Strings**
Python allows various types of operations on strings, including arithmetic, relational, logical operations, loops, and membership operations.



### Arithmetic Operations

1. **Concatenation**: Combining two strings using the `+` operator.
2. **Repetition**: Repeating a string multiple times using the `*` operator.

```python
# Concatenation
print("Hello" + " " + "World")

# Repetition
print("*" * 50)
```

### Relational Operations

Relational operations compare strings lexicographically (based on the Unicode value of each character).

1. **Equality (`==`)**

2. **Inequality (`!=`)**

3. **Greater than (`>`)**

4. **Less than (`<`)**

5. **Greater than or equal to (`>=`)**

6. **Less than or equal to (`<=`)**

```python
"delhi" == "mumbai"
```

```python
"mumbai" > "pune"
```

🤔 **Explanation:** <br>
The output of `"mumbai" > "pune"` is `False` because string comparison in Python is done lexicographically, meaning it compares strings based on the Unicode values of their characters. It follows the same rules as alphabetical ordering in dictionaries.

**Lexicographical Comparison:**<br>

When comparing `"mumbai"` and `"pune"`, Python compares each character from the start until it finds a difference:

1. Compare the first characters: `'m'` and `'p'`.
2. The Unicode value of `'m'` (109) is less than the Unicode value of `'p'` (112).

Since `'m'` is less than `'p'`, `"mumbai"` is considered less than `"pune"` lexicographically, and thus `"mumbai" > "pune"` evaluates to `False`.

```python
"Pune" > "pune"
```

🤔 **Explanation:** <br>

The output of `"Pune" > "pune"` is `False` because of the way Python handles lexicographical comparisons, which are based on the Unicode values of the characters. In Unicode, uppercase letters have lower values than lowercase letters. When comparing `"Pune"` and `"pune"`, Python compares each character from the start until it finds a difference:

1. Compare the first characters: `'P'` and `'p'`.
2. The Unicode value of `'P'` (80) is less than the Unicode value of `'p'` (112).

Since `'P'` is less than `'p'`, `"Pune"` is considered less than `"pune"` lexicographically, and thus `"Pune" > "pune"` evaluates to `False`.



### Logical Operations

```python
"hello" and "world"
```

🤔 **Explanation:** <br>

The expression `"hello" and "world"` in Python evaluates to `"world"` due to the way the `and` logical operator functions. In Python, the `and` operator returns the first falsy value it encounters or the last value if all values are truthy. Both `"hello"` and `"world"` are non-empty strings, and non-empty strings are considered truthy in Boolean contexts. Since both operands are truthy, the `and` operator returns the last value, which is `"world"`. This behavior ensures that if any operand in the chain is falsy, it stops evaluating further and returns that falsy value; otherwise, it returns the last operand, which in this case is `"world"`.

```python
"hello" or "world"
```

🤔 **Explanation:**<br>

The expression `"hello" or "world"` in Python evaluates to `"hello"` due to the behavior of the `or` logical operator. In Python, the `or` operator returns the first truthy value it encounters or the last value if all are falsy. In this expression, `"hello"` is a non-empty string, which is considered truthy in a Boolean context. Because `"hello"` is truthy, the `or` operator does not need to evaluate the second operand, `"world"`, and immediately returns `"hello"`. This mechanism ensures that the `or` operator returns the first truthy value found, which in this case is `"hello"`, making the entire expression evaluate to `"hello"`.

```python
"" and "world"
```

🤔 **Explanation:**<br>

The expression `"" and "world"` in Python evaluates to `""` because of how the `and` logical operator works. In Python, the `and` operator returns the first falsy value it encounters or the last value if all operands are truthy. In this case, `""` is an empty string, which is considered falsy in a Boolean context. As a result, when evaluating the expression `"" and "world"`, Python immediately encounters the falsy value `""` and returns it without evaluating the second operand, `"world"`. This behavior ensures that the `and` operator stops at the first falsy value and returns it, making the entire expression evaluate to `""`.

```python
"" or "world"
```

🤔 **Explanation:** <br>

The expression `"" or "world"` in Python evaluates to `"world"` due to the behavior of the `or` logical operator. In Python, the `or` operator returns the first truthy value it encounters or the last value if all operands are falsy. In this case, `""` is an empty string, which is considered falsy in a Boolean context. When evaluating the expression `"" or "world"`, Python first encounters the falsy value `""` and then moves on to evaluate the next operand, `"world"`, which is a non-empty string and therefore truthy. Since `"world"` is the first truthy value in the expression, the `or` operator returns `"world"`. This demonstrates how the `or` operator ensures that the first truthy value is returned, making the entire expression evaluate to `"world"`.

```python
print(not "")
print(not "hello")
```

### Loops on Strings
Strings in Python are iterable, meaning you can loop through each character in a string using different types of loops. This allows you to perform operations or process each character individually.

```python
for i in "hello":
    print(i)
```

```python
# # Quick question
# # Guess the output of this code cell
# for i in "delhi":
#     print("pune")
```

### Membership Operations

Membership operations in Python allow you to check whether a substring exists within another string. These operations are done using the `in` and `not in` operators.

```python
"D" in "Delhi"
```

```python
"d" not in "Delhi"
```

## **String Functions**


### Common Functions

Python provides several built-in functions to perform operations on strings. Four commonly used functions are `len()`, `max()`, `min()`, and `sorted()`.

1. `len()`: The `len()` function returns the number of characters in a string.

2. `max()`: The `max()` function returns the character with the highest Unicode value from the string. If the string is empty, it raises a `ValueError`.

3. `min()`: The `min()` function returns the character with the lowest Unicode value from the string. If the string is empty, it raises a `ValueError`.
   
4. `sorted()`: The `sorted()` function returns a list of characters from the string, sorted in ascending order based on their Unicode values.

```python
myStr = "Hello"

print(len(myStr))
print(max(myStr))
print(min(myStr))
print(sorted(myStr)) # Sorted in ascending order
print(sorted(myStr, reverse=True)) # Sorted in descending order
```

### Capitalize/Title/Upper/Lower/Swapcase

1. `capitalize()`: Capitalizes the first character of the string and converts all other characters to lowercase.

2. `title()`: Capitalizes the first character of each word in the string and converts all other characters to lowercase.

3. `upper()`: Converts all characters in the string to uppercase.

4. `lower()`: Converts all characters in the string to lowercase.

5. `swapcase()`: Swaps the case of all characters in the string; converts uppercase characters to lowercase and vice versa.

```python
myStr = "HeLLo WoRLd"

print(myStr.capitalize())
print(myStr.title())
print(myStr.upper())
print(myStr.lower())
print(myStr.swapcase())
```

### Count/Find/Index

1. `count()`: Returns the number of occurrences of a specified substring in the string.

2. `find()`: Returns the lowest index of the specified substring if it is found in the string; otherwise, it returns `-1`.

3. `index()`: Returns the lowest index of the specified substring if it is found in the string; otherwise, it raises a `ValueError`.

```python
myStr = "My name is Krishnagopal"

print(myStr.count("i"))
print(myStr.find("is"))
print(myStr.index("is"))
```

### Endswith/Startswith
1. `endswith()`: Checks if the string ends with a specified suffix. It returns `True` if the string ends with the suffix, and `False` otherwise.

2. `startswith()`: Checks if the string starts with a specified prefix. It returns `True` if the string starts with the prefix, and `False` otherwise.

```python
print(myStr.startswith("My"))
print(myStr.startswith("name"))

print(myStr.endswith("l"))
print(myStr.endswith("is"))
```

### Format
The `format()` method in Python is used to format strings by embedding values within them. It allows you to insert and format values into a string using curly braces `{}` as placeholders.

```python
first_name = "Krishnagopal"
last_name = "Halder"

"My name is {} {}".format(first_name, last_name)
```

### isalnum/isalpha/isdigit/isidentifier
1. `isalnum()`: Returns `True` if all characters in the string are alphanumeric (i.e., letters and digits) and there is at least one character; otherwise, it returns `False`.

2. `isalpha()`: Returns `True` if all characters in the string are alphabetic (i.e., letters) and there is at least one character; otherwise, it returns `False`.

3. `isdigit()`: Returns `True` if all characters in the string are digits and there is at least one character; otherwise, it returns `False`.

4. `isidentifier()`: Returns `True` if the string is a valid identifier (i.e., it starts with a letter or an underscore and consists of letters, digits, or underscores), and `False` otherwise.

```python
print("Alpha123".isalnum())
print("Krishnagopal".isalpha())
print("3214".isdigit())
print("fist_name".isidentifier())
print("1first_name".isidentifier())
```

### Split/Join
Here’s a brief overview of the `split()` and `join()` string methods in Python:

1. `split()`
- **Description**: Splits a string into a list of substrings based on a specified delimiter (separator). By default, it splits on any whitespace and removes extra whitespace.
- **Syntax**: `string.split(separator, maxsplit)`
  - `separator` (optional): The delimiter on which to split the string. If not specified, whitespace is used.
  - `maxsplit` (optional): The maximum number of splits to perform. The default value `-1` means "all occurrences."

2. `join()`
- **Description**: Joins elements of an iterable (such as a list or tuple) into a single string, with a specified separator between each element.
- **Syntax**: `separator.join(iterable)`
  - `separator`: The string used as a separator between elements of the iterable.
  - `iterable`: The iterable whose elements will be joined into a single string.



```python
myStr = "My name is Krishnagopal Halder"

print(myStr.split())
print(myStr.split("is"))
```

```python
print(" ".join(['My', 'name', 'is', 'Krishnagopal', 'Halder']))
print("-".join(['My', 'name', 'is', 'Krishnagopal', 'Halder']))
```

### Replace
The `replace()` method is used to replace occurrences of a specified substring with another substring within a string. 

- **Usage**: This method is useful for replacing parts of a string with another string.
- **Syntax**: `string.replace(old, new, count)`
  - `old`: The substring to be replaced.
  - `new`: The substring to replace the old substring with.
  - `count` (optional): The maximum number of occurrences to replace. If not specified, all occurrences are replaced.

```python
myStr.replace("Krishnagopal Halder", "Krishna")
```

### Strip
The `strip()` method is used to remove leading and trailing whitespace (or specified characters) from a string.

- **Usage**: This method is useful for cleaning up strings by removing unwanted whitespace or specific characters from both ends of the string.
- **Syntax**: `string.strip([chars])`
  - `chars` (optional): A string specifying the set of characters to be removed. If not specified, the method removes whitespace by default.

- **Variants**

1. **`strip()`**: Removes characters from both ends of the string.
2. **`lstrip()`**: Removes characters from the left end of the string.
3. **`rstrip()`**: Removes characters from the right end of the string.

```python
"Krishnagopal Halder         ".strip()
```

## **Exercises**

```python
# Find the length of a given string without using the len() function
```

```python
# Extract username from a given email. 
# Eg if the email is halder24krishnagopal@gmail.com 
# then the username should be halder24krishnagopal
```

```python
# Count the frequency of a particular character in a provided string. 
# Eg 'hello how are you' is the string, the frequency of h in this string is 2.
```

```python
# Write a program which can remove a particular character from a string.
```

```python
# Write a program that can check whether a given string is palindrome or not.
# abba
# malayalam
```

```python
# Write a program to count the number of words in a string without split()
```

```python
# Write a python program to convert a string to title case without using the title()
```

```python
# Write a program that can convert an integer to string.
```

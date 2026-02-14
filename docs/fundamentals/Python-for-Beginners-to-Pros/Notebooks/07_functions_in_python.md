[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Notebooks/07_functions_in_python.ipynb)

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

# **Functions in Python**

In Python, functions are blocks of reusable code that perform a specific task. Functions help to organize code, avoid repetition, and improve modularity. They allow you to define a piece of logic once and use it multiple times throughout your program.

<img src="https://data-flair.training/blogs/wp-content/uploads/sites/2/2018/01/Python-Functions.jpg" width="50%">

**Abstraction** and **decomposition** are two key concepts in programming that help manage complexity, especially when using functions.

1. **Abstraction**:
Abstraction is the process of **hiding the complex details** of a task and exposing only the essential features or functionalities. In the context of functions, abstraction means creating a function that performs a specific task without requiring the user of the function to know how it works internally.

2. **Decomposition**:
Decomposition is the process of **breaking down a large problem** into smaller, more manageable parts. In the context of functions, it involves splitting a complex task into simpler sub-tasks, where each sub-task can be handled by a separate function. This approach makes code easier to understand, test, and maintain.


## **Syntax of a Function**
The syntax of a function in Python includes the function header, the function body, and optionally a return statement. Here's a breakdown of the structure:

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220721172423/51.png" width="50%">

```python
# Write a simple function to check whether a number is even or not
def is_even(i):
    """
    Optional docstrings which tells us about the function, things like what inputs are required, what will the function return.

    Args:
        i (int): The input number to check.

    Returns:
        bool: True if the number is even, False otherwise.
    """

    if i % 2 == 0:
        return True
    else:
        return False

# Check the function (Function calling)
is_even(3)
```

<!-- The syntax of a function in Python includes the function header, the function body, and optionally a return statement. Here's a breakdown of 

Explanation:<br>
- **`def`**: This keyword is used to define a new function.
- **`function_name`**: This is the name you give to the function. It follows the same naming conventions as variables (e.g., no spaces, cannot start with a number).
- **`parameters`**: These are the inputs the function can take (optional). If there are multiple parameters, they are separated by commas.
- **`:`**: A colon marks the end of the function header and indicates that the function body will follow.
- **Docstring** (Optional): A triple-quoted string that describes the purpose of the function. It helps document the code.
- **Function body**: This is the indented block of code that contains the logic of the function. It can have any number of statements.
- **`return`**: This keyword is used to return a value from the function (optional). If omitted, the function returns `None` by default. -->

```python
# Print the ddcumentation of a function
print(is_even.__doc__)
```

## **Two Point of Views**

- Function Creator's View:
  - **Clarity and modularity**: Design the function to be reusable, well-organized, and easy to understand.
  - **Robustness**: Handle different input cases and ensure reliable performance with proper documentation.

- Function User's View:
  - **Simplicity**: The function should be easy to use with clear inputs and outputs.
  - **Reliability**: The user expects the function to work correctly without needing to know its internal workings.

```python
# Modify the function to handle invalid data types gracefully without throwing an error
def is_even(i):
    """Optional docstrings which tells us about the function, things like what inputs are required, what will the function return

    Args:
        i (int): The input number to check.

    Returns:
        bool: True if the number is even, False otherwise.
    """

    if type(i) == int:
        if i % 2 == 0:
            return True
        else:
            return False
    else:
        print("Are you Mad?")

# Check the function (Function calling)
is_even("y")
```

## **Parameters** vs **Arguments**

- **Parameters** are the variables listed inside the function's definition. They act as placeholders for the values that the function expects to receive.
  
  Example:
  ```python
  def greet(name):  # 'name' is a parameter
      print(f"Hello, {name}!")
  ```

- **Arguments** are the actual values passed to the function when it is called. These values are assigned to the corresponding parameters.

  Example:
  ```python
  greet("Alice")  # "Alice" is an argument
  ```

**Key Differences:**
- **Parameters** are used when defining a function.
- **Arguments** are the actual values supplied to the function during execution.


## **Types of Arguments**

In Python, there are several types of arguments that can be passed to a function. These include:

1. **Default Arguments**:
A default argument in Python is an argument that takes a default value if no value is provided for it when the function is called. If the caller does not provide a corresponding argument during the function call, the default value is used.

```python
# Example
def greet(name, age=18): # Default argument for 'age'
    print(f"Hello, {name}! You are {age} years old.")

greet("Aditi")  # Only 'name' is provided; 'age' uses its default value of 18
```

2. **Positional Arguments**: Positional arguments are arguments that are passed to a function in the correct positional order, meaning the first argument is assigned to the first parameter, the second argument to the second parameter, and so on. The order in which arguments are passed matters, and they must match the function parameters' positions.

```python
# Example
def greet(name, age):
    print(f"Hello, {name}! You are {age} years old.")

# Calling the function with positional arguments
greet("Aditi", 18)  # Output: Hello, Aditi! You are 18 years old.
```

3. **Keyword Arguments**: Keyword arguments are arguments that are passed to a function using the name of the parameter explicitly, allowing you to assign values to specific parameters regardless of their order in the function definition. This makes the function call more readable and flexible.

```python
# Example
def greet(name, age):
    print(f"Hello, {name}! You are {age} years old.")

# Calling the function with keyword arguments
greet(age=18, name="Aditi")
```

<!-- #region -->
## **`*args`** vs **`**kwargs`**

In Python, `*args` and `**kwargs` are used in function definitions to allow the function to accept an arbitrary number of arguments. They are particularly useful when you don't know how many arguments will be passed to the function.

**1. `*args` (Non-keyword variable-length arguments)**:
- `*args` allows a function to accept any number of positional arguments. These arguments are passed as a tuple inside the function.


- Key Points about `*args`:
  - You can pass any number of positional arguments.
  - Inside the function, `args` is treated as a tuple.
  - The `*` is required before the parameter name (`args` can be replaced with any name, e.g., `*numbers`).
<!-- #endregion -->

```python
# Write a function to calculate average of n numbers
def average(*args):

    sum_of_nums = 0
    for i in args:
        sum_of_nums += i

    return sum_of_nums / len(args)

# Call the function
average(2, 4, 6)
```

**2. `**kwargs` (Keyword variable-length arguments)**:
- `**kwargs` allows a function to accept any number of keyword arguments. These arguments are passed as a dictionary inside the function.

- Key Points about `**kwargs`:
  - You can pass any number of keyword arguments.
  - Inside the function, `kwargs` is treated as a dictionary where the keys are parameter names, and the values are the corresponding arguments.
  - The `**` is required before the parameter name (`kwargs` can be replaced with any name, e.g., `**info`).


```python
# Write a function to display the information of a person
def display_info(**kwargs):

    for key, value in kwargs.items():
        print(f"{key}: {value}")

display_info(name="Aditi", age=18, city="Bonn", country="Germany")
```

**3. Combining `*args` and `**kwargs`:**
   - We can also write a function that combines normal parameters, `*args`, and `**kwargs`. But in this case, the parameters must follow a specific order:

     1. **Regular Parameters**: Required positional parameters come first.
     2. **`*args`**: Allows for additional positional arguments (as a tuple) and must be placed after regular parameters.
     3. **Default Parameters**: Parameters with default values follow `*args`.
     4. **`**kwargs`**: Allows for additional keyword arguments (as a dictionary) and must be placed last.


```python
# Example
def employee_info(name, age, *args, company="Google", **kwargs):
    """
    Display the information about an employee.

    Args:
        name (str): The employee's name
        age (int): The employee's age
        *args: Additional positional arguments (e.g., hobbies).
        company (str, optional): Company name. Defaults to "Google".
        **kwargs: Additional keyword arguments (e.g., address, skills).

    Returns:
        None
    """

    # Print the normal parameters
    print(f"Name: {name}")
    print(f"Age: {age}")

    # Print the default parameter
    print(f"Company: {company}")

    # Print additional positional arguments
    if args:
        print("Hobbies:")
        for i in args:
            print(f"\t{i}")

    # Print additional keyword arguments
    if kwargs:
        for key, value in kwargs.items():
            print(f"{key}: {value}")

# Call the function
employee_info("Sundar Pichai", 48, "Watching Cricket", "Listening Music", 
              email="pichaiS@gmail.com", phone=123456789)
```

## **How Functions are Executed in Memory?**

When a function runs, here’s what happens in simple terms:

1. **Function Call**: When you call a function, a special "box" (called a stack frame) is created to hold everything the function needs, like its inputs (parameters) and temporary variables.

2. **Memory Use**: The function uses two main areas in memory:
   - **Stack**: For storing its inputs and local variables (small, short-term things).
   - **Heap**: For bigger, longer-lasting things like objects or large data (if needed).

3. **Running the Function**: The computer follows the instructions in the function, using the data in the stack to do its job.

4. **Finishing Up**: When the function is done, it gives back a result (if needed) and removes the "box" (stack frame) it used, so the memory is freed up for the next function.

5. **Memory Cleanup**: If the function made any big objects or data, the computer may clean them up later to free up space (in some languages, this happens automatically). 

So, the function gets its own space to work, does its job, and then cleans up when it’s done. To visualize the concept, please try this website: https://pythontutor.com/

```python
# Try the tool with this function
# Write a simple function to check whether a number is even or not
def is_even(i):

    if i % 2 == 0:
        return True
    else:
        return False

# Check the function (Function calling)
is_even(3)
```

## **Functions with No Return Statement**
When a function doesn't have a `return` statement, it still runs and does its tasks, but it **doesn’t send any value back** to where it was called. In most programming languages, such functions automatically return a special value, like `None` in Python, or they just complete without returning anything in other languages like C or Java.

**What Happens:**<br>
1. **Execution**: The function runs and performs its operations just like any other function.
2. **No Return**: If there’s no `return` statement, the function simply finishes after its last instruction.
3. **Default Return**: In languages like Python, the function automatically returns `None`, meaning "nothing." In other languages, it may not return any value at all.

```python
L = [1, 2, 3]
print(L.append(4))
print(L)
```

## **Variable Scope**

Variable scope refers to the part of a program where a variable can be accessed or used. There are different scopes that determine where a variable is visible and how long it exists in memory.

**Types of Variable Scope:**

1. **Local Scope**:
   - Variables defined inside a function or block are **local variables**.
   - They can only be accessed within that function or block.
   - Once the function finishes, the local variables are deleted from memory.

2. **Global Scope**:
   - Variables defined outside all functions or blocks are **global variables**.
   - They can be accessed from anywhere in the program, including inside functions.
   - Global variables exist throughout the program’s execution.


```python
# Example-1
def g(y): # 'y' is a local variable
    print(x)
    print(x+1)

x = 5 # 'x' is a global variable
g(x)
print(x)
```

- **`x`** is a **global variable**, accessible both inside and outside the function `g()`. It remains unchanged after the function call.
- **`y`** is a **local variable** to the function `g()`, only available inside the function, but it is not used in this code.
- Inside the function, `x` refers to the global `x`, since there's no local `x` defined in the function.

```python
# Exmaple-2
def f(y):
    x = 1 # here, 'x' is a local variable
    x += 1
    print(x)

x = 5 # here, 'x' is a global varialble. There is no relation with the variable 'x' in the local scope
f(x)
print(x)
```

- **`x` (global)**: Defined outside the function (`x = 5`), accessible globally but not affected by the function `f()`.
- **`x` (local)**: Defined inside the function (`x = 1`), exists only within the function `f()` and is independent of the global `x`.

```python
# # Example-3
# def h(y):
#     x += 1

# x = 5
# h(x)
# print(x)
```

- **`x` (global)**: Defined outside the function (`x = 5`), intended to be accessible globally.
- **`x` (local)**: Inside the function `h(y)`, there is an attempt to modify `x` using `x += 1`, but no local `x` is defined. This leads to an error because the function tries to modify the global `x` without declaring it as global inside the function. 

```python
# Example-3
def h(y):
    global x
    x += 1

x = 5
h(x)
print(x)
```

- **`x` (global)**: The variable `x` is defined globally with `x = 5` and is accessed and modified inside the function `h(y)` using the `global` keyword. 
- **`x` (local)**: Inside the function `h(y)`, `x` is treated as a global variable because of the `global` keyword. This allows the function to modify the global `x`.

```python
# Example-4
def f(x):
    x = x + 1
    print("in f(x): x =", x)
    return x

x = 3
z = f(x)
print("in main program scope: z =", z)
print("in main program scope: x =", x)
```

## **Nested Functions**

A **nested function** is a function that is defined inside another function. In Python (and many other programming languages), you can define functions inside other functions to create more modular and maintainable code. The inner function is encapsulated within the outer function, meaning it is only accessible within the outer function and not from the outside scope.

**Key points about nested functions:**
1. **Scope**: The inner function can access variables from the outer function, but the outer function cannot access the inner function's variables unless returned explicitly.
2. **Encapsulation**: Nested functions are useful for hiding helper functions that are not meant to be accessed from outside.
3. **Closures**: If the inner function captures the variables of the outer function and keeps them in memory after the outer function has finished executing, this is called a closure.

```python
# Example-1
def outer_function():
    def inner_function():
        print("inside inner function")
    inner_function()
    print("inside outer function")

outer_function()
```

```python
# Example-2
def g(x):
    def h():
        x = "abc"
    x = x + 1
    print("in g(x): x =", x)
    h()
    return x

x = 3
z = g(x)
```

```python
# Example-3
def g(x):
    def h(x):
        x = x+1
        print("in h(x): x =", x)
    x = x + 1
    print("in g(x): x =", x)
    h(x)
    return x

x = 3
z = g(x)
print("in main program scope: x =", x)
print("in main program scope: z =", z)
```

## **Functions are 1st Class Citizens**

In Python, functions are first-class citizens, meaning they are treated like any other object/datatype, such as integers, strings, or lists. This concept allows functions to be used flexibly and passed around within the code just like other objects.

```python
# type and id
def square(num):
    return num**2

print(type(square))
print(id(square))
```

```python
# reassign
x = square
print(id(x))
print(x(3))
```

```python
# deleting a function
del square
```

```python
# storing
L = [1, 2, 3, 4, square]
L[-1](3)
```

```python
S = {square}
S # Function is immutable as it can be stored in a set
```

```python
# returning a function
def f():
    def x(a, b):
        return a+b
    return x

val = f()(3, 4)
print(val)
```

```python
# function as argument
def func_a():
    print("inside func_a")

def func_b(z):
    print("inside func_b")
    return z()

print(func_b(func_a))
```

## **Benefits of using a Function**

1. **Code Reusability**: Define once, use multiple times, reducing redundancy.
2. **Modularity**: Break complex tasks into smaller, manageable parts.
3. **Abstraction**: Hide internal details; users only need to know what the function does.
4. **Code Organization**: Group related tasks, making code easier to read and maintain.
5. **Simplified Debugging/Testing**: Test individual functions independently.
6. **Reduces Code Duplication**: Avoid repeated code by reusing functions.
7. **Improved Maintainability**: Easier to update logic in one place.
8. **Supports Recursion**: Solve problems by calling a function within itself.
9. **Facilitates Functional Programming**: Enables higher-order functions and functional programming patterns.


## **Lambda Function**
A **lambda function** in Python is a small, anonymous function defined using the `lambda` keyword. Unlike regular functions defined with `def`, lambda functions are typically used for short, simple operations and are defined in a single line.

**Key characteristics:**
- **Anonymous**: Lambda functions don't require a name.
- **Single Expression**: They can contain only a single expression (no statements or multiple lines).
- **Syntax**: 
  ```python
  lambda arguments: expression
  ```

  <img src="https://media.licdn.com/dms/image/v2/D4D12AQEchdPSA86FdA/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1687841327729?e=1736380800&v=beta&t=e95ll6lmE1Jh5zgkQRiTmvMQjoKlLOOvo-7Awb_uw6Y" width="50%"> 

```python
# x -> x^2
square = lambda x: x**2
square(2)
```

```python
# x, y -> x+y
sum_ot_two_numbers = lambda x, y: x + y
sum_ot_two_numbers(5, 2)
```

### **Difference between Lambda vs Normal Function**
Here’s a concise comparison between **lambda** and **normal (def)** functions:

1. **Syntax**:
   - **Lambda**: `lambda args: expression` (one-liner)
   - **Normal**: `def function_name(args):` (multi-line)

2. **Name**:
   - **Lambda**: Anonymous, unless assigned to a variable.
   - **Normal**: Always named.

3. **Use Case**:
   - **Lambda**: For short, simple tasks.
   - **Normal**: For complex, multi-step logic.

4. **Expressions**:
   - **Lambda**: Single expression only.
   - **Normal**: Multiple statements allowed.

5. **Readability**:
   - **Lambda**: Compact, less readable for complex tasks.
   - **Normal**: Clearer, especially for longer code.

6. **Return**:
   - **Lambda**: Implicit return.
   - **Normal**: Explicit `return`.

Lambda functions are best for short, throwaway tasks; normal functions are better for more complex or reusable code. Lambda functions are generally used with Higher Order Function (HOF).

```python
# Check if a string has 'a'
check_a = lambda s: 'a' in s
check_a("Hello")
```

```python
# Check a number whether it is odd or even
odd_or_even = lambda num: "even" if num % 2 == 0 else "odd"
odd_or_even(3)
```

### **Higher Order Functions**
A **Higher-Order Function** is a function that either:

1. **Takes one or more functions as arguments**, or
2. **Returns a function** as its result.

In other words, higher-order functions operate on other functions, treating them as "first-class citizens" (objects that can be passed, returned, or assigned).

```python
# Example
def square(num):
    return num**2

def cube(num):
    return x**3

# Higher Order Function (HOF)
def transform(f, L):
    output = []
    for i in L:
        output.append(f(i))

    print(output)

L = [1, 2, 3, 4, 5]
transform(square, L)
```

```python
transform(lambda x: x**3, L)
```

<!-- #region -->
### **Map**
The **`map()`** function in Python applies a given function to each item of an iterable (like a list or tuple) and returns an iterator (or map object) with the results.

**Syntax:**
```python
map(function, iterable)
```
- **function**: The function to apply to each element.
- **iterable**: The iterable (list, tuple, etc.) whose items will be passed into the function.
<!-- #endregion -->

```python
# Square of the items of a list
list(map(lambda x: x**2, [1, 2, 3, 4, 5, 6]))
```

```python
# Odd/Even labelling of list items
L = [1, 2, 3, 4, 5]
list(map(lambda num: "even" if num%2 == 0 else "odd", L))
```

```python
# Fetch names from a list of dict
users = [
    {
        "name": "Rahul",
        "age": 15,
        "gender": "male"

    },
    {
        "name": "Aditi",
        "age": 14,
        "gender": "female"
    },
    {
        "name": "Sneha",
        "age": 13,
        "gender": "female"
    }
]

list(map(lambda user: user["name"], users))
```

<!-- #region -->
### **Filter**
The **`filter()`** function in Python is used to filter elements from an iterable (like a list or tuple) based on a condition defined in a function. It returns an iterator containing only the elements for which the function returns `True`.

**Syntax:**
```python
filter(function, iterable)
```
- **function**: A function that tests each element and returns `True` or `False`.
- **iterable**: The iterable (list, tuple, etc.) to be filtered.
<!-- #endregion -->

```python
# Number greater than 5
L = [3, 4, 5, 6, 7]

list(filter(lambda x: x > 5, L))
```

```python
# Fetch fruits starting with 'a'
fruits = ["apple", "guava", "cherry"]

list(filter(lambda x: x.startswith("a"), fruits))
```

<!-- #region -->
### **Reduce**

The **`reduce()`** function in Python is used to apply a binary function (a function that takes two arguments) cumulatively to the items of an iterable, reducing the iterable to a single accumulated value.

It is part of the **`functools`** module, so you need to import it first.

**Syntax:**
```python
from functools import reduce
reduce(function, iterable, [initializer])
```
- **function**: A function that takes two arguments and performs an operation (like addition, multiplication, etc.).
- **iterable**: The iterable (list, tuple, etc.) whose elements are processed.
- **initializer** (optional): A value that is used to start the accumulation. If provided, it's used as the initial value; otherwise, the first element of the iterable is used.

**How it works:**
- **First**: It applies the function to the first two elements of the iterable.
- **Next**: It applies the function to the result of the previous operation and the next element of the iterable.
- **Repeat**: It continues this process until only one result remains.
<!-- #endregion -->

```python
# Sum of all numbers
from functools import reduce

reduce(lambda x, y: x+y, [1, 2, 3, 4, 5])
```

```python
# Find min
reduce(lambda x, y: min(x, y), [23, 11, 45, 10, 5])
```

```python
reduce(lambda x, y: x if x<y else y, [23, 11, 45, 10, 5])
```

```python
# Find max
reduce(lambda x, y: x if x>y else y, [23, 11, 45, 10, 5])
```

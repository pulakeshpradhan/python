[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Notebooks/04_time_complexity.ipynb)

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

# **Time Complexity**
Time complexity is a computational concept that describes the amount of time an algorithm takes to complete as a function of the length of the input. It provides an upper bound on the time an algorithm will take to run and is typically expressed using Big O notation, which describes the worst-case scenario.


## **What is Efficiency in Programming?**

Efficiency in programming refers to how effectively an algorithm or code performs in terms of resource usage. This includes both time efficiency (how fast the code runs) and space efficiency (how much memory the code uses). Efficient code performs its intended task using the least amount of computational resources possible, balancing both time and space constraints.

Key aspects of efficiency in programming include:

1. **Time Efficiency**:
   - **Execution Time**: How quickly an algorithm completes its task. This is often evaluated using time complexity analysis (e.g., Big O notation).
   - **Response Time**: How long it takes for a system or application to respond to a user action or request.

2. **Space Efficiency**:
   - **Memory Usage**: The amount of memory an algorithm or program uses during execution. This is often evaluated using space complexity analysis.
   - **Data Structures**: Choosing the most appropriate data structures can significantly impact memory usage and access times.
  
Efficient programming is crucial for creating high-performance applications, especially in environments with limited resources or where performance is critical, such as real-time systems, large-scale data processing, and high-frequency trading systems.

<!-- #region vscode={"languageId": "plaintext"} -->
## **Techniques to Measure Time Complexity**
Measuring time efficiency involves various techniques to evaluate how quickly an algorithm or program executes. Here are some common techniques:

### 1. **Measuring Time to Execute**
Manually measure the execution time of a code block using built-in functions. Measuring time to execute is a common approach to evaluating the performance of an algorithm or piece of code. However, this method has several limitations and potential problems:

1. **System Dependence**
   - **Hardware Variability**: Execution time can vary significantly across different hardware configurations (e.g., different CPUs, memory speeds).
   - **Operating System Variability**: Different operating systems or even different states of the same OS can affect performance due to background processes and system load.

2. **Environment Variability**
   - **Background Processes**: Other running applications or processes can consume CPU time and memory, leading to inconsistent measurements.
   - **Network Latency**: For programs that depend on network resources, network latency and bandwidth fluctuations can affect execution time.

3. **Measurement Overhead**
   - **Timing Overhead**: The act of measuring execution time itself can introduce overhead, especially for very short code segments, making the measurement less accurate.
   - **Profiling Overhead**: Using profiling tools can add significant overhead, distorting the actual performance characteristics of the code.
<!-- #endregion -->

```python
import time

start = time.time()

for i in range(100):
    print(i)

end = time.time()

print("Time taken:", end-start, "seconds.")
```

### 2. **Counting Operations Involved**

Manually count the number of key operations (e.g., comparisons, assignments) in the code to estimate its time complexity.

- **Advantages:**

  1. **Hardware Independence**: Provides a measure of algorithm efficiency that is not affected by the underlying hardware or system   environment.
   
  2. **Theoretical Insight**: Offers a clear understanding of the algorithm's behavior and complexity, helping to predict performance for different input sizes.
   
  3. **Granularity**: Allows for detailed analysis of specific parts of the algorithm, identifying potential bottlenecks and areas for optimization.
   
  4. **Predictability**: Enables consistent and repeatable results, as the operation count is deterministic for a given input.

- **Disadvantages:**

  1. **Complexity in Implementation**: Manually counting operations can be tedious and error-prone, especially for complex algorithms.

  2. **Simplification Assumptions**: Often focuses on a single type of operation (e.g., comparisons, swaps), which may oversimplify the actual performance characteristics.

  3. **Ignoring Constant Factors**: Does not account for constant time operations or overheads, which can be significant in practical scenarios.
   
  4. **No Direct Execution Time**: Provides an abstract measure of complexity but does not translate directly to actual execution time, which is also influenced by factors like memory access patterns and cache performance.

```python
# Function to convert Celsius (°C) value to its Fahrenheit (°F)
def cel_to_farh(c):
    return c * 9/5 + 32 # total 3 operations


# Function to compute sum of digits in all numbers from 1 to n
def mysum(n):
    sum = 0 # 1st operation
    for i in range(n+1): # 3n operations
        sum += i
    return sum # total 1 + 3n operations
```

### 3. **Orders of Growth**

Orders of growth are used to describe the asymptotic behavior of an algorithm's time or space complexity as the input size increases. This concept helps in comparing the efficiency of different algorithms, particularly for large input sizes. 

**Goals:** <br> The goals of understanding and analyzing orders of growth in algorithm design and analysis are fundamental to creating efficient and scalable software. Here are the key goals:
1. Want to evaluate program's efficiency when the input is very big.
2. Want to express the growth of program's run time as input size grows.
3. Want to put an upper bound on growth - as tight as possible.
4. Do not need to be precise: "order of" not "exact" growth.
5. We will look at the largest factors in run time (which section of the program will take the longest to run?)
6. Thus, generally we want tight upper bound on growth, as function of the size input, in worst case.


#### **Exact Steps vs O()**
- Computes factorial
- Number of steps
- Worst case asymptotic complexity:
  - ignore additive constants
  - ignore multiplicative constants

```python
# Write a function to calculate the factorial of any given number
def factorial(n):
    """assuming n is an integer >= 0"""
    result = 1 # 1 operation
    while n > 1: # 1 operation
        result *= n # 2 operations
        n -= 1 # 2 operations
    return result # 1 operation

# Total operations: 5n + 2
```

```python

```

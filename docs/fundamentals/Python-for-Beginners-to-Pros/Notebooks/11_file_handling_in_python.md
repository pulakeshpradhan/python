[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/fundamentals/Python-for-Beginners-to-Pros/Notebooks/11_file_handling_in_python.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
---

# File Handling in Python
In this notebook, we'll explore basic file handling operations as well as using the `os` and `glob` modules.


## 1. Basic File Handling in Python

```python
# Writing to a file
with open('sample.txt', 'w') as f:
    f.write('Hello, World!\n')
    f.write('Welcome to file handling in Python.\n')
print("File written successfully.")
```

```python
# Reading from a file
with open('sample.txt', 'r') as f:
    content = f.read()
print("File content:")
print(content)
```

```python
# Appending to a file
with open('sample.txt', 'a') as f:
    f.write('Appending a new line!\n')
print("Line appended successfully.")
```

```python
# Reading the file again
with open('sample.txt', 'r') as f:
    content = f.read()
print("Updated file content:")
print(content)
```

## 2. Using the `os` Module

```python
import os

# Get current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")
```

```python
# List all files and directories in the current directory
entries = os.listdir(cwd)
print("Directory contents:")
for entry in entries:
    print(entry)
```

```python
# Check if a file or directory exists
file_exists = os.path.exists('sample.txt')
print(f"Does 'sample.txt' exist? {file_exists}")
```

```python
# Create a new directory
new_dir = 'test_dir'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
    print(f"Directory '{new_dir}' created.")
else:
    print(f"Directory '{new_dir}' already exists.")
```

```python
# Rename a file
os.rename('sample.txt', 'renamed_sample.txt')
print("File renamed to 'renamed_sample.txt'.")
```

```python
# Remove a file
os.remove('renamed_sample.txt')
print("File 'renamed_sample.txt' removed.")
```

```python
# Remove the directory
os.rmdir('test_dir')
print(f"Directory '{new_dir}' removed.")
```

## 3. Using the `glob` Module

```python
import glob

# Create some files for demonstration
for i in range(3):
    with open(f'file_{i}.txt', 'w') as f:
        f.write(f"This is file {i}\n")

print("Demo files created: file_0.txt, file_1.txt, file_2.txt")
```

```python
# Find all .txt files
txt_files = glob.glob('*.txt')
print("List of .txt files:")
print(txt_files)
```

```python
# You can also use wildcard patterns
file_1_pattern = glob.glob('file_1.*')
print("Files matching 'file_1.*':")
print(file_1_pattern)
```

```python
# Cleanup: remove demo files
for file in txt_files:
    os.remove(file)
print("Demo files removed.")
```

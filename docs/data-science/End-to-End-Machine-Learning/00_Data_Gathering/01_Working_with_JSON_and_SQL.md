[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/00_Data_Gathering/01_Working_with_JSON_and_SQL.ipynb)

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# **Working with JSON/SQL**


## **Import Required Libraries**

```python
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
```

## **Working with JSON**
JSON (JavaScript Object Notation) is a lightweight data interchange format that is easy for humans to read and write, and easy for machines to parse and generate. It is commonly used for transmitting data between a server and a web application, as well as for configuration files, APIs, and various other purposes.

JSON is designed to be a language-independent data format, meaning it can be used with any programming language that has the capability to parse and generate JSON data. It is often used in web development for sending and receiving structured data, as it's more concise and easier to work with than XML.

JSON data is represented as a collection of key-value pairs, where each key is a string and each value can be a string, number, boolean, object, array, or null. JSON objects are enclosed in curly braces {}, and each key-value pair is separated by a colon. JSON arrays are ordered lists of values and are enclosed in square brackets [].


**Example:**

```JSON
{
  "name": "John Doe",
  "age": 30,
  "isStudent": false,
  "hobbies": ["reading", "swimming", "gardening"]
}

```


### **Read a Local JSON File**

```python
json_path = r"D:\Coding\Datasets\train.json"
pd.read_json(json_path)
```

### **Read a JSON File from URL**

```python
json_url = r"https://raw.githubusercontent.com/LearnWebCode/json-example/master/animals-1.json"
pd.read_json(json_url)
```

## **Working with SQL**
SQL stands for Structured Query Language. It's a domain-specific programming language used for managing and manipulating relational databases. SQL is used to create, modify, and query databases to store, retrieve, and manipulate data in a structured manner. It provides a standardized way to interact with databases, regardless of the specific database management system (DBMS) being used.


### **Read sql Data**
mysql.connector is a Python library used to connect and interact with MySQL databases. It provides a Pythonic way to work with MySQL databases by allowing you to execute SQL queries, manage connections, and handle the results. This library is commonly used to integrate MySQL databases with Python applications.

```python
# !pip install mysql.connector
```

```python
import mysql.connector
```

```python
conn =mysql.connector.connect(host="localhost", user="root", password="", database="world")
```

```python
pd.read_sql_query("SELECT * FROM city", conn)
```

```python
pd.read_sql_query("SELECT * FROM city WHERE CountryCode LIKE 'IND'", conn)
```

```python
pd.read_sql_query("SELECT * FROM country WHERE LifeExpectancy > 60", conn)
```

```python
# Storing the dataframe in a variable
df = pd.read_sql_query("SELECT * FROM country WHERE LifeExpectancy > 60", conn)
df.head()
```

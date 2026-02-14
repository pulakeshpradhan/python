[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/00_Data_Gathering/03_Fetching_Data_using_Web_Scrapping.ipynb)

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

# **Fetching Data using Web Scrapping**
Web scraping is the process of automatically extracting information and data from websites. It involves writing code to retrieve specific pieces of content, such as text, images, tables, or other structured data, from web pages. Web scraping is commonly used to gather data for various purposes, such as research, analysis, data mining, automation, and more.

Web scraping can range from simple tasks, like retrieving the title of an article from a news website, to more complex tasks, like extracting financial data from multiple web pages for analysis. However, it's important to note that while web scraping can be a powerful tool, it should be used ethically and responsibly. Some websites have terms of use that prohibit or restrict scraping, and improperly scraping data or overloading a website's server with requests can have legal and ethical implications.


## **Import Required Libraries**

```python
import pandas as pd
import requests
from bs4 import BeautifulSoup
```

## **Fetch Webpage from the Server**

```python
web_url = r"https://www.ambitionbox.com/list-of-companies?page=2"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36"}
webpage = requests.get(web_url, headers=headers).text
```

```python
# Creating an object of the webpage using BeautifulSoup Class
soup = BeautifulSoup(webpage, "lxml")
```

## **Extract Companies Name from the URL**

```python
# Create an empty list of companies
companies = []
for i in range(0, 20):
    company = soup.find_all("h2")[i].text.strip()
    companies.append(company)
```

```python
# Print all the companies name
companies
```

```python
# Print the length of companies
len(companies)
```

## **Extract Rating of All Companies**

```python
# Create an empty list of ratings
ratings = []
for i in range(0, 20):
    rating = soup.find_all("span", class_="companyCardWrapper__companyRatingValue")[i].text.strip()
    ratings.append(rating)
```

```python
# Print all the ratings
ratings
```

```python
# Print the length of ratings
len(ratings)
```

## **Extract Content of All Companies**

```python
# Create a list of contents
contents = []
for i in range(0, 20):
    content = soup.find_all("span", class_="companyCardWrapper__interLinking")[i].text.strip()
    contents.append(content)
```

```python
# Print all the content
print(contents)
```

```python
# Print the length of contents
len(contents)
```

## **Write a Function to Automate the Workflow**

```python
# Creating a function to fetch the data for each page
def fetchData(soup):

    companies = soup.find_all("div", class_="companyCardWrapper__companyDetails")

    # Extracting the data of the companies
    companyNames = []
    ratings = []
    contents = []
    industries = []
    employees = []
    ctypes = []
    age = []
    hq = []

    for i in range(0, 20):
        company = companies[i]

        # Getting the name of the company
        company_name = company.find("h2", class_="companyCardWrapper__companyName").text.strip()
        companyNames.append(company_name)

        # Extracting the ratings
        rating = company.find("span", class_="companyCardWrapper__companyRatingValue").text.strip()
        ratings.append(rating)

        # Extracting the contents
        content = company.find("span", class_="companyCardWrapper__interLinking").text.strip()
        # Convert the content into a list
        content = content.split(sep=" | ")

        # Extracting the industry
        industry = content[0]
        industries.append(industry)

        # Extracting the employees
        employee = ""
        ctype = ""
        old = ""
        headquarter = ""

        for j in content[1:]:
            if "Employees" in j:
                employee = j
            elif "old" in j:
                old = j
            elif "more" in j:
                headquarter = j.split(sep=" ")[0:-2]
                headquarter = " ".join(headquarter)
            else:
                ctype = j

        if len(employee) > 0:
            employees.append(employee[0:-10])
        else:
            employees.append("NaN")

        if len(old) > 0:
            age.append(old[0:-4])
        else:
            age.append("NaN")

        if len(headquarter) > 0:
            hq.append(headquarter)
        else:
            hq.append("NaN")

        if len(ctype) > 0:
            ctypes.append(ctype)
        else:
            ctypes.append("NaN")
    
    # Create a dataframe
    columns = {"name": companyNames, "rating": ratings, "industry": industries,
               "employee": employees, "type": ctypes, "age": age, "headquarter": hq}
    df = pd.DataFrame(columns)
    
    return df
```

```python
# Checking the fetchData function
fetchData(soup)
```

## **Fetch the Data for all the Webpages**

```python
# Store the headers in a variable
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36"}

# Create a blank dataframe
final_df = pd.DataFrame()

# Storing data in the final dataframe for 500 pages
for i in range(1, 500):
    web_url = "https://www.ambitionbox.com/list-of-companies?page={}".format(i)
    webpage = requests.get(web_url, headers=headers).text
    
    soup = BeautifulSoup(webpage, "lxml")
    
    df = fetchData(soup)
    final_df = pd.concat([final_df, df], ignore_index=True)
```

```python
final_df
```

## **Export the Final Dataframe as a CSV**

```python
output_path = r"D:\\Coding\\Datasets\\"
file_name = "company_web_scrapping_data.csv"
final_df.to_csv(output_path+file_name)
```

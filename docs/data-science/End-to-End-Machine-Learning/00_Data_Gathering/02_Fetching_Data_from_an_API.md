[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/data-science/End-to-End-Machine-Learning/00_Data_Gathering/02_Fetching_Data_from_an_API.ipynb)

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

# **Fetching Data from an API**
Fetching data from APIs using Python involves utilizing the requests library to make HTTP requests to a remote server and retrieve data. APIs (Application Programming Interfaces) provide a standardized way for different software applications to communicate and exchange information. 


## **Import Required Libraries**

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
```

## **Setup the API**

```python
url = "https://air-quality.p.rapidapi.com/history/airquality"

querystring = {"lon":"87.0624","lat":"23.1645"}

headers = {
	"X-RapidAPI-Key": "YOUR_API_KEY",
	"X-RapidAPI-Host": "air-quality.p.rapidapi.com"
}
```

## **Send the Request**

```python
response = requests.get(url, headers=headers, params=querystring)

print(response.json())
```

## **Convert JSON into Dataframe**

```python
# Converting json into dataframe
df = pd.DataFrame(response.json()["data"])
```

```python
df.head()
```

```python
df.shape
```

## **Preparing the Data**

```python
# Extracting only required columns
required_columns = ["timestamp_local", "aqi", "no2", "so2", "o3", "co", "pm10", "pm25"]
weather_data = df[required_columns]
```

```python
weather_data.head()
```

```python
weather_data.info()
```

```python
# Converting 'timestamp_local' column into datetime object
weather_data["timestamp_local"] = pd.to_datetime(weather_data["timestamp_local"])
```

```python
weather_data.info()
```

```python
# Sorting the data by timestamp
weather_data.sort_values(by="timestamp_local", inplace=True)
```

```python
# Set the 'timestamp_local' as index
weather_data.set_index("timestamp_local", inplace=True)
weather_data.head()
```

```python
weather_data.shape
```

## **Plot the Data**

```python
# Plot a line graph
plt.figure(figsize=(15, 5))
sns.lineplot(data=weather_data, x=weather_data.index, y=weather_data.aqi,
             marker="o", label="AQI")
sns.lineplot(data=weather_data, x=weather_data.index, y=weather_data.no2, 
             marker="s", label="NO2")
sns.lineplot(data=weather_data, x=weather_data.index, y=weather_data.so2, 
             marker="D", label="SO2")
plt.xlabel("Time", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.title("Weather Data Time Series", fontsize=16)
plt.show()
```

```python
# Plot a heatmap based on correlation of weather constituents
sns.heatmap(weather_data.corr(), annot=True, cmap="PRGn")
plt.title("Correlation between different Weather Constituents", weight="500")
plt.show()
```

## **Save the Dataframe as CSV**

```python
output_path = r"D:\Coding\Datasets\bankura_one_day_weather.csv"
weather_data.to_csv(output_path)
```

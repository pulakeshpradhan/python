[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Geospatial_Data_Science_with_Python/03_Exploratory_Spatial_Data_Analysis/01_Exploratory_Data_Visualization.ipynb)

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

# **Exploratory Data Visualization**
Exploratory data visualization is the process of creating visual representations of data to gain insights, discover patterns, and identify relationships within a dataset. It involves the use of various graphical techniques and plots to explore the characteristics and structure of the data. The primary goal of exploratory data visualization is to understand the data and generate hypotheses or ideas for further analysis.

Common techniques used in exploratory data visualization include:

1. **Scatter plots:** Scatter plots display the relationship between two variables by plotting data points on a two-dimensional graph. They are useful for identifying correlations or patterns between variables.

2. **Histograms:** Histograms provide a visual representation of the distribution of a single variable. They group data into bins or intervals and display the frequency or count of observations within each bin.

3. **Box plots:** Box plots, also known as box-and-whisker plots, summarize the distribution of a variable by displaying quartiles, outliers, and the range of the data.

4. **Bar charts:** Bar charts are used to compare categories or groups by representing the values of different variables as rectangular bars. They are commonly used for categorical data.

**Spatial Data Visualization:**<br>
Spatial data visualization focuses on representing geographic or spatial data in visual form. It involves the use of maps, spatial plots, and other geospatial visualizations to explore and communicate patterns, relationships, and distributions across geographic areas.

Common techniques used in spatial data visualization include:

1. **Choropleth maps:** Choropleth maps use different colors or shading to represent the intensity or magnitude of a variable across regions or areas. They are effective for displaying data at an aggregated level, such as population density or election results by region.

2. **Scatter plots on maps:** Scatter plots can be overlaid on maps to visualize the relationship between variables at specific geographic locations. This allows for the exploration of spatial patterns or clusters.

3. **Heatmaps:** Heatmaps can be used in spatial data visualization to represent the density or intensity of events or occurrences across a geographic area. They provide a visual depiction of hotspots or areas of concentration.

4. **Cartograms:** Cartograms distort the size or shape of regions on a map based on a variable of interest, allowing for the visualization of relative magnitudes or proportions.


## **01. Import Required Libraries**

```python
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

## **0.2 Setting Up the Current Working Directory**

```python
# Checking the current working directory
os.getcwd()
```

```python
# Changing the current working directory
file_path = r"D:\Coding\Git Repository\Geospatial_Data_Science_with_Python\Datasets"
os.chdir(file_path)
csv_path = file_path + "\\CSVs"
shp_path = file_path + "\\Shapafiles"
```

```python
# Checking the new current working directory
os.getcwd()
```

## **03. Reading the Data**


**Dataset Description:**<br>
This is the dataset used in the second chapter of Aurélien Géron's recent book 'Hands-On Machine learning with Scikit-Learn and TensorFlow'. It serves as an excellent introduction to implementing machine learning algorithms because it requires rudimentary data cleaning, has an easily understandable list of variables and sits at an optimal size between being to toyish and too cumbersome.

The data contains information from the 1990 California census. So although it may not help you with predicting current housing prices like the Zillow Zestimate dataset, it does provide an accessible introductory dataset for teaching people about the basics of machine learning.

**Content:**<br>
1. longitude: A measure of how far west a house is; a higher value is farther west
2. latitude: A measure of how far north a house is; a higher value is farther north
3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
4. totalRooms: Total number of rooms within a block
5. totalBedrooms: Total number of bedrooms within a block
6. population: Total number of people residing within a block
7. households: Total number of households, a group of people residing within a home unit, for a block
8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
9. medianHouseValue: Median house value for households within a block (measured in US Dollars)
10. oceanProximity: Location of the house w.r.t ocean/sea

```python
# Reading the housing.csv data with pandas
housing = pd.read_csv(csv_path + "\\housing.csv")
# Checking the name of the columns
housing.columns
```

```python
# Checking the first 5 rows of the data
housing.head()
```

```python
# Checking the shape of the dataframe
housing.shape
```

## **04. Conducting Exploratory Data Analysis (EDA)**

```python
# Checking the non-null values in each column
housing.info()
```

```python
# Cleaning the data
housing.dropna(inplace=True)
```

```python
# Checking the shape of the dataframe
housing.shape
```

```python
# Checking the value counts of the ocean_proximity column
housing["ocean_proximity"].value_counts()
```

```python
# Definining a dictionary to encode the values of ocean_proximity column from string to int
ocean_proximity_dict = {"ISLAND": 0, "NEAR BAY": 1, "NEAR OCEAN": 2, "INLAND": 3, "<1H OCEAN": 4}
# Encoding the ocean_proximity column
encoded_ocean_proximity = housing["ocean_proximity"].replace(ocean_proximity_dict)
```

```python
# Creating a copy of housing dataframe
housing_copy = housing.copy()
```

```python
# Setting the encoded values of ocean_proximity column
housing_copy["ocean_proximity"] = encoded_ocean_proximity
# Checking the first 5 rows of the new housing_copy dataframe
housing_copy.head()
```

```python
# Checking the non-null values in each column of the new housing_copy dataframe
housing_copy.info()
```

```python
# Dropping the rows with null values
housing_copy.dropna(inplace=True)
# Resetting the index
housing_copy.reset_index(inplace=True, drop=True)
```

```python
# Checking the final dataframe
housing_copy.head()
```

```python
# Checking the dataframe information
housing_copy.info()
```

```python
# Describing the dataframe
housing_copy.describe()
```

```python
# Create a visual representation of the data
housing_copy.hist(bins=50, figsize=(20, 18))
```

## **05. Exploratory Spatial Data Analysis (ESDA)**

```python
# Converting the pandas dataframe to a geopandas dataframe
housing_gdf = gpd.GeoDataFrame(housing_copy, geometry=gpd.points_from_xy(housing_copy.longitude, housing_copy.latitude, crs=4326))
```

```python
# Checking the CRS of the geodataframe
housing_gdf.crs
```

**Geoplot:**<br>
Geoplot is a Python library that provides a high-level interface for creating a wide range of geographical visualizations using matplotlib. It is built on top of geopandas, which is a powerful library for working with geospatial data. Geoplot simplifies the process of creating maps, enabling users to quickly generate various types of plots to visualize spatial data.

Geoplot offers a set of plot types that are commonly used in geographic data analysis, such as choropleth maps, kernel density estimation (KDE) plots, cartograms, and spatial lags. These plots can be easily customized to suit specific visualization requirements.

One of the key features of geoplot is its ability to work seamlessly with geopandas. Geopandas provides data structures to work with geospatial data, such as points, lines, and polygons, and allows users to perform spatial operations on them. Geoplot takes advantage of these data structures and operations, enabling users to create geospatial visualizations by leveraging the power of geopandas.

Geoplot provides an intuitive API that allows users to create plots with just a few lines of code. It integrates well with Jupyter notebooks, making it ideal for interactive data exploration and analysis. The library supports various map projections and provides tools for handling geographic coordinate reference systems (CRS).

```python
# Importing geoplot library
import geoplot.crs as gcrs
import geoplot as gplt
```

```python
# Plotting the housing_gdf using geoplot
ax = gplt.webmap(housing_gdf, projection=gcrs.WebMercator())
gplt.pointplot(housing_gdf, ax=ax, marker=".")
```

```python
# Loading the USA States shapefile
usa_states = gpd.read_file(shp_path + "\\tl_2021_us_state.zip")
# Checking the first five rows of the geodataframe
usa_states.head()
```

```python
# Filtering the California geometry
california = usa_states["geometry"][usa_states["NAME"]=="California"]
# Checking the CRS of the California geometry
california.crs
```

```python
# Changing the CRS to Web Mercator (4326)
california = california.to_crs(4326)
# Plotting the California map
california.plot(color="white", edgecolor="black", linewidth=0.5)
```

```python
# Checking the housing_gdf dataframe
housing_gdf.head()
```

**Heatmap:**<br>
A heatmap is a graphical representation of data where values are depicted as colors on a two-dimensional grid. It is particularly useful for visualizing the distribution and intensity of data points across different categories or dimensions.

In a heatmap, each cell of the grid represents a combination of two variables, typically displayed along the X and Y axes. The color of each cell corresponds to the value of a third variable, often referred to as the "intensity" or "magnitude" of the data. The colors used in the heatmap are usually chosen to represent a continuous spectrum, ranging from low to high values.

```python
# Creating a heatmap of point locations
ax = gplt.kdeplot(housing_gdf,
                  fill=True, 
                  cmap="Reds", 
                  clip=california.geometry, 
                  projection=gcrs.WebMercator()
                  )
# Plotting the California polygon on top of the heatmap
gplt.polyplot(california, ax=ax, zorder=1)
```

```python
# Creating a point plot to display the spatial variation of median_house_value
ax = gplt.pointplot(housing_gdf,
                    hue="median_house_value",
                    scale="median_house_value",
                    cmap="Reds",
                    legend=True,
                    projection=gcrs.WebMercator()
                    )
# Plotting the Califonia polygon on top of the point plot
gplt.polyplot(california, ax=ax, zorder=0)
ax.set_title("Median House Value in California, USA in 1990")
```

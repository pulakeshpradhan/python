[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/geospatial/Geospatial_Data_Science_with_Python/02_Exploring_Geospatial_Packages/05_Shapely.ipynb)

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

# **Shapely**
**Author: Krishnagopal Halder**

Shapely is a powerful Python library for geometric operations and manipulations. It provides a convenient and intuitive interface for working with geometric objects such as points, lines, polygons, and more. Developed by Sean Gillies, Shapely is built on top of the widely-used GEOS (Geometry Engine - Open Source) library, which enables robust and efficient geometric computations.

Shapely is particularly useful in the field of spatial analysis and geospatial applications. It allows you to perform a wide range of operations on geometric objects, including spatial relationships, geometric transformations, spatial predicates, and spatial measurements. Whether you need to calculate distances between points, determine if two polygons intersect, or simplify a complex geometry, Shapely provides the tools to accomplish these tasks.


## **01. Importing Required Libraries**

```python
import shapely.geometry
import shapely.wkt
import matplotlib.pyplot as plt
```

## **02. Creating Shapely Geometry from WKT**
In Shapely, WKT stands for "Well-Known Text," which is a standard text-based representation format for geometric objects. Shapely provides support for parsing WKT strings into Shapely geometry objects and generating WKT representations from existing geometries.

To work with WKT in Shapely, you need to import the shapely.wkt module.

We have a WKT string "Polygon((0 0, 0 -5, 10 -5, 10 0, 0 0))", which represents a Polygon with five points. We use wkt.loads() to parse the WKT string and create a Shapely Ploygon geometry object.

By leveraging the WKT format, you can easily create Shapely geometries from text-based representations and vice versa. This is particularly useful when you need to store or exchange geometric data in a standard format that is independent of specific programming languages or software packages.

```python
# Defining WKT strings to represent a polygon, a line string and a point
poly_wkt = "Polygon((0 0, 0 -5, 10 -5, 10 0, 0 0))"
line_wkt = "LineString(0 0, 10 0, 10 -5)"
point_wkt = "Point(0 0)"
# Parsing the WKT strings to create Shapely geometry objects
poly = shapely.wkt.loads(poly_wkt)
line = shapely.wkt.loads(line_wkt) 
point = shapely.wkt.loads(point_wkt)
```

```python
# Printing the Polygon Shapely geometry
print(poly)
print(type(poly))
poly
```

```python
# Printing the LineString Shapely geometry
print(line)
print(type(line))
line
```

```python
# Printing the Point Shapely geometry
print(point)
print(type(point))
point
```

## **03. Creating Shapely Geometry from GeoJSON**
The GeoJSON object is represented by a dictionary in Python, containing a "type" field and a "coordinates" field. The "type" field indicates the type of geometry, such as "Point," "Polygon," or "MultiLineString." The "coordinates" field holds the coordinates defining the geometry.

Shapely supports various GeoJSON geometry types, including Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, and GeometryCollection.

```python
from shapely.geometry import Polygon, LineString, Point
```

```python
# Defining GeoJSON to represent a polygon, a line string and a point
poly_geojson = {"type": "Polygon",
               "coordinates": [[[0, 0],
                               [0, -5],
                               [10, -5],
                               [10, 0],
                               [0, 0]]]}
line_geojson = {"type": "LineString",
               "coordinates": [[[0, 0],
                               [10, 0],
                               [10, -5]]]}
point_geojson = {"type": "Point",
                "coordinates": [[0, 0]]}
```

```python
# Parsing the GeoJSON to create Shapely geometry objects
poly_GJ = Polygon([tuple(i) for i in poly_geojson["coordinates"][0]])
line_GJ = LineString([tuple(i) for i in line_geojson["coordinates"][0]])
point_GJ = Point(tuple(point_geojson["coordinates"][0]))
```

```python
# Printing the Polygon Shapely geometry
print(poly_GJ)
print(type(poly_GJ))
poly_GJ
```

```python
# Printing the Line Shapely geometry
print(line_GJ)
print(type(line_GJ))
line_GJ
```

```python
# Printing the Point Shapely geometry
print(point_GJ)
print(type(point_GJ))
point_GJ
```

## **04. Creating Shapely Geometry from List of Coordinates**
To create a Shapely geometry from a list of coordinates in Python using the Shapely library, you can utilize the appropriate geometry constructor provided by Shapely. The specific constructor you use depends on the type of geometry you want to create, such as Point, LineString, Polygon, etc.

By passing the list of coordinates to the Shapely geometry constructor, you can easily create a Shapely geometry object.

```python
# Defining Coordinate lists to represent a polygon, a line string and a point
poly_coords = [(0, 0), (0, -5), (10, -5), (10, 0), (0, 0)]
line_coords = [(0, 0), (10, 0), (10, -5)]
point_coords = [(0, 0)]
```

```python
# Parsing the coordinate lists to create Shapely geometry objects
poly_list = Polygon(poly_coords)
line_list = LineString(line_coords)
point_list = Point(point_coords)
```

```python
# Printing the Polygon Shapely geometry
print(poly_list)
print(type(poly_list))
poly_list
```

```python
# Printing the LineString Shapely geometry
print(line_list)
print(type(line_list))
line_list
```

```python
# Printing the Point Shapely geometry
print(point_list)
print(type(point_list))
point_list
```

## **05. Creating MultiPolygon, MultiPoint and MultiLineString**

```python
from shapely.geometry import MultiPolygon, MultiPoint, MultiLineString
```

**MultiPolygon:**<br>
A MultiPolygon is a collection of Polygon geometries. Each Polygon represents a closed area defined by a boundary consisting of linear rings. The MultiPolygon can be used to represent complex areas composed of multiple polygons. For instance, a region with multiple islands can be represented as a MultiPolygon.

```python
# Creating MultiPolygon from WKT
multiPoly_wkt = "MultiPolygon(((0 0, 0 -10, 10 -10, 10 0, 0 0), (3 -2, 6 -8, 8 -2, 3 -2)))"
multi_poly_wkt = shapely.wkt.loads(multiPoly_wkt)
print(multi_poly_wkt)
print(type(multi_poly_wkt))
multi_poly_wkt
```

```python
# Creating MultiPolygon from List of Coordinates
polygon1 = Polygon([(0, 0), (0, -10), (10, -10), (10, 0), (0, 0)])
polygon2 = Polygon([(3, -2), (6, -8), (8, -2), (3, -2)])
multiPoly_list = [polygon1, polygon2]
multi_poly_list = MultiPolygon(multiPoly_list)
print(multi_poly_list)
print(type(multi_poly_list))
multi_poly_list
```

**MultiPoint:** <br>
A MultiPoint is a collection of Point geometries. Each Point represents a specific location in the 2D space. The MultiPoint can be used to represent multiple discrete points or a set of spatially related locations.

```python
# Creating MultiPoint from WKT
multiPoint_wkt = "MultiPoint((0 0), (5 5), (0 10))"
multi_point_wkt = shapely.wkt.loads(multiPoint_wkt)
print(multi_point_wkt)
print(type(multi_point_wkt))
multi_point_wkt
```

```python
# Creating MultiPoint from List of Coordinates
point1 = Point((0, 0))
point2 = Point((5, 5))
point3 = Point((0, 10))
multiPoint_list = [point1, point2, point3]
multi_point_list = MultiPoint(multiPoint_list)
print(multi_point_list)
print(type(multi_point_list))
multi_point_list
```

**MultiLineString:** <br>
A MultiLineString is a collection of LineString geometries. Each LineString represents a sequence of connected line segments. The MultiLineString can be used to represent complex linear features composed of multiple LineString segments. For example, a road network with multiple interconnected road segments can be represented as a MultiLineString.

```python
# Creating MultiLineString from WKT
multiline_wkt = "MultiLineString((0 0, 5 0, 5 -5), (10 0, 10 5, 15 5))"
multi_line_wkt = shapely.wkt.loads(multiline_wkt)
print(multi_line_wkt)
print(type(multi_line_wkt))
multi_line_wkt
```

```python
# Creating MultiLineString from List of Coordinates
line1 = LineString([(0, 0), (5, 0), (5, -5)])
line2 = LineString([(10, 0), (10, 5), (15, 5)])
multiLine_list = [line1, line2]
multi_line_list = MultiLineString(multiLine_list)
print(multi_line_list)
print(type(multi_line_list))
multi_line_list
```

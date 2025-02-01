# Scaling and Reprojecting Geospatial Data

## Overview
This Python library provides utility functions for preprocessing geospatial data, including scaling raster values, handling coordinate reference systems (CRS), reprojecting geospatial data, and masking no-data values. It is designed to facilitate geospatial analysis and machine learning applications that work with raster and vector data.

## Features
- **Z-Score Scaling**: Standardizes raster data by centering around zero and scaling by standard deviation.
- **Min-Max Scaling**: Scales pixel values to a fixed range (e.g., [0, 1] or [-1, 1]).
- **Retrieve CRS**: Extracts the coordinate reference system from vector and raster data.
- **Compare CRS**: Checks if raster and vector data share the same CRS.
- **Reproject Data**: Converts raster or vector data to a specified CRS.
- **Mask No-Data Values**: Identifies and masks undefined pixel values in raster data.

## Installation
Ensure you have the required dependencies installed before using this library:
```bash
pip install numpy geopandas rasterio rioxarray xarray pyproj
```

## Usage
### 1. Scaling Raster Data
```python
import numpy as np
from scaling_and_reproject import Z_score_scaling, Min_Max_Scaling

data = np.array([[10, 20, 30], [40, 50, 60]])
z_scaled = Z_score_scaling(data)
minmax_scaled = Min_Max_Scaling(data)
```

### 2. Getting and Comparing CRS
```python
import geopandas as gpd
from scaling_and_reproject import get_crs, compare_crs

vector_data = gpd.read_file("vector.shp")
raster_data = rasterio.open("raster.tif")

vector_crs = get_crs(vector_data)
raster_crs = get_crs(raster_data)
comparison = compare_crs(raster_data, vector_data)
```

### 3. Reprojecting Data
```python
from scaling_and_reproject import reproject_data

target_crs = "EPSG:4326"
reprojected_vector = reproject_data(vector_data, target_crs)
```

### 4. Masking No-Data Values
```python
import rioxarray as rxr
from scaling_and_reproject import mask_raster_data

raster = rxr.open_rasterio("raster.tif")
masked_raster = mask_raster_data(raster)
```

## License
This project is licensed under the MIT License.

## Author
[Your Name] – [Your Email or GitHub Profile]

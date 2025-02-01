# Scaling and Reprojecting Geospatial Data

## Overview
This Python library provides utility functions for preprocessing geospatial data, including scaling raster values, handling coordinate reference systems (CRS), reprojecting geospatial data, and masking no-data values. It is designed to facilitate geospatial analysis and machine learning applications that work with raster and vector data.

## Features 
- **Data Normalization**
  - Z-Score Scaling: Standardizes raster data by centering around zero and scaling by standard deviation.
  - Min-Max Scaling: Scales pixel values to a fixed range (e.g., [0, 1] or [-1, 1]).
- **CRS Management**
  - Retrieve CRS: Extracts the coordinate reference system from vector and raster data.
  - Compare CRS: Checks if raster and vector data share the same CRS.
- **Reprojection**
  - Vector data reprojection (GeoPandas)
  - Raster reprojection (Rasterio/rioxarray)
- **Data Masking**
  - No-data value handling
  - Cross-format support (numpy/xarray)

## Installation
Ensure you have the required dependencies installed before using this library:
```bash
pip install numpy geopandas rasterio rioxarray xarray pyproj
```

## Usage
### 1. Data Scaling
```python
import numpy as np
from scaling_and_reproject import Z_score_scaling, Min_Max_Scaling

data = np.array([[10, 20, 30], [40, 50, 60]])
z_scaled = Z_score_scaling(data)
minmax_scaled = Min_Max_Scaling(data)
```

### 2. CRS Management
```python
import geopandas as gpd
import rasterio
from scaling_and_reproject import get_crs, compare_crs

vector = gpd.read_file("data.shp")
raster = rasterio.open("image.tif")

print(get_crs(vector))  # EPSG:4326
print(compare_crs(raster, vector))  # CRS comparison results
```

### 3. Reprojection
```python
import rasterio
import xarray as xr
from scaling_and_reproject import reproject_data

# Vector reprojection
reprojected_vector = reproject_data(vector, "EPSG:3857")

# Raster reprojection (Rasterio)
with rasterio.open("input.tif") as src:
    array, metadata = reproject_data(src, "EPSG:32633")

# Xarray reprojection
da = xr.open_rasterio("image.tif")
reprojected_da = reproject_data(da, "EPSG:4326")
```

### 4. Data Masking
```python
import xarray as xr
import rasterio
from scaling_and_reproject import mask_raster_data

# Rasterio workflow
with rasterio.open("data.tif") as src:
    data = src.read(1)
    masked, profile = mask_raster_data(data, src.profile)

# rioxarray workflow
da = xr.open_rasterio("data.tif")
masked_da = mask_raster_data(da)
```

## License
This project is licensed under the MIT License.

## Author
[Your Name] – [Your Email or GitHub Profile]

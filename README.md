# GeoPre: Geospatial Data Processing Toolkit  
**GeoPre** is a Python library designed to streamline common geospatial data operations, offering a unified interface for handling raster and vector datasets. It simplifies preprocessing tasks essential for GIS analysis, machine learning workflows, and remote sensing applications.


### Key Features  
- **Data Scaling**:  
  - Normalization (Z-Score) and Min-Max scaling for raster bands.  
  - Prepares data for ML models while preserving geospatial metadata.  

- **CRS Management**:  
  - Retrieve and compare Coordinate Reference Systems (CRS) across raster (Rasterio/Xarray) and vector (GeoPandas) datasets.  
  - Ensure consistency between datasets with automated CRS checks.  

- **Reprojection**:  
  - Reproject vector data (GeoDataFrames) and raster data (Rasterio/Xarray) to any target CRS.  
  - Supports EPSG codes, WKT, and Proj4 strings.  

- **No-Data Masking**:  
  - Handle missing values in raster datasets (NumPy/Xarray) with flexible masking.  
  - Integrates seamlessly with raster metadata for error-free workflows.  


### Supported Data Types  
- **Raster**: NumPy arrays, Rasterio `DatasetReader`, Xarray `DataArray` (via rioxarray).  
- **Vector**: GeoPandas `GeoDataFrame`.  


### Benefits of GeoPre  
- **Unified Workflow**: Eliminates boilerplate code by providing consistent functions for raster and vector data.  
- **Interoperability**: Bridges gaps between GeoPandas, Rasterio, and Xarray, ensuring smooth data transitions.  
- **Robust Error Handling**: Automatically detects CRS mismatches and missing metadata to prevent silent failures.  
- **Efficiency**: Optimized reprojection and masking operations reduce preprocessing time for large datasets.  
- **ML-Ready Outputs**: Scaling functions preserve data structure, making outputs directly usable in machine learning pipelines.  


Ideal for researchers and developers working with geospatial data, **GeoPre** enhances productivity by standardizing preprocessing steps and ensuring compatibility across diverse geospatial tools.


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

## Contributing

1. **Fork the repository**  
   
   Click the "Fork" button at the top-right of this repository to create your copy.
   
2. **Create your feature branch**  
   ```bash
   git checkout -b feature/your-feature
   
3. **Commit changes**  
   ```bash
   git commit -am 'Add some feature'
   
4. **Push to branch**  
   ```bash
   git push origin feature/your-feature

5. **Open a Pull Request**
   
   Navigate to the Pull Requests tab in the original repository and click "New Pull Request" to submit your changes.

   
## License
This project is licensed under the MIT License. See LICENSE for more information.


## Author
[Your Name] – [Your Email or GitHub Profile]

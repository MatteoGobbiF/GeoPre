# GeoPre: Geospatial Data Processing Toolkit  
[![PyPI version](https://img.shields.io/pypi/v/geopre.svg)](https://pypi.org/project/geopre/)
[![License](https://img.shields.io/github/license/MatteoGobbiF/GeoPre.svg)](https://github.com/MatteoGobbiF/GeoPre/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/geopre.svg)](https://pypi.org/project/geopre/)

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

- **Cloud Masking**:  
  - Identify and mask clouds in Sentinel-2 and Landsat imagery.  
  - Supports multiple methods: QA bands, scene classification layers (SCL), probability bands, and OmniCloudMask AI-based detection.  
  - Optionally mask cloud shadows for improved accuracy.  

- **Water Masking** (NEW in v0.3.0):  
  - Mask water bodies in Sentinel-2 and Landsat imagery using the NDWI index.  
  - Tries to automatically detects relevant bands (Green/NIR). 

- **Band Stacking**:  
  - Stack multiple raster bands from a folder into a single multi-band raster for analysis.  
  - Supports automatic band detection and resampling for different resolutions.  


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
GeoPre is available on [PyPI](https://pypi.org/project/geopre/) and can be installed with:
```bash
pip install geopre
```
This will automatically install all required dependencies.



## Usage
### 1. Data Scaling
#### `Z-Score Scaling`
**Description**:This method centers the data around zero by subtracting the mean and dividing by the standard deviation, which is useful for machine learning models sensitive to outliers 
and can standardize a band of pixel values for clustering/classification.

**Parameters**:
- data (numpy.ndarray): Input array to normalize.
        
**Returns**:
- numpy.ndarray: Standardized data with mean 0 and standard deviation 1.

#### `Min_Max_Scaling`
**Description**: This method scales the pixel values to a fixed range, typically [0, 1] or [-1, 1]. Ideal when you want to preserve the relative range of values. 
For GeoTIFF image values (e.g., 0 to 65535), scale them to [0, 1].

**Parameters**:
- data (numpy.ndarray): Input array to normalize.

**Returns**:
- numpy.ndarray: Scaled data with values between 0 and 1, or -1 and 1.
      
#### Example:
```python
import numpy as np
import geopre as gp

data = np.array([[10, 20, 30], [40, 50, 60]])
z_scaled = gp.Z_score_scaling(data)
minmax_scaled = gp.Min_Max_Scaling(data)
```

### 2. CRS Management

#### `get_crs`
**Description**: Retrieve CRS from geospatial data objects.

**Parameters**:
- data: GeoPandas GeoDataFrames (vector), Rasterio DatasetReaders (raster) or Xarray DataArrays with rio accessor (raster)

**Returns**:
- pyproj.CRS: Coordinate reference system or None if undefined

#### `compare_crs`
**Description**: Compare CRS between raster and vector datasets.

**Parameters**:
- raster_obj (DatasetReader/xarray.DataArray): Raster data source.
- vector_gdf (gpd.GeoDataFrame): Vector data source.
 
**Returns**:

**dict**: Comparison results with keys:
- raster_crs: Formatted CRS string
- vector_crs: Formatted CRS string  
- same_crs: Boolean comparison result
- error: Exception message if any

#### Example:
```python
import geopandas as gpd
import rasterio
import geopre as gp

vector = gpd.read_file("data.shp")
raster = rasterio.open("image.tif")

print(gp.get_crs(vector))  # EPSG:4326
print(gp.compare_crs(raster, vector))  # CRS comparison results
```

### 3. Reprojection
#### `reproject_data`
**Description**: Reproject geospatial data to target CRS.

**Parameters**:
- data: GeoDataFrames (vector reprojection), or Rasterio datasets (returns array + metadata), or Xarray objects (rioxarray reprojection) 
- target_crs: CRS to reproject to (EPSG code/WKT/proj4 string)

**Returns**:
- Reprojected data in format matching input type

#### Example:
```python
import rasterio
import xarray as xr
import geopre as gp

# Vector reprojection
reprojected_vector = gp.reproject_data(vector, "EPSG:3857")

# Raster reprojection (Rasterio)
with rasterio.open("input.tif") as src:
    array, metadata = gp.reproject_data(src, "EPSG:32633")

# Xarray reprojection
da = xr.open_rasterio("image.tif")
reprojected_da = gp.reproject_data(da, "EPSG:4326")
```

### 4. No-Data Masking
#### `mask_raster_data`
**Description**: Mask no-data values in raster datasets. Handles both rasterio (numpy) and rioxarray (xarray) workflows.

**Parameters**:
- data: Raster data (numpy.ndarray or xarray.DataArray)
- profile: Rasterio metadata dict (required for numpy arrays)
- no_data_value: Override for metadata's nodata value
- return_mask: Whether to return boolean mask

**Returns**:
- Masked data array. For numpy inputs, returns tuple:(masked_array, profile). For xarray, returns DataArray.

#### Example:
```python
import xarray as xr
import rasterio
import geopre as gp

# Rasterio workflow
with rasterio.open("data.tif") as src:
    data = src.read(1)
    masked, profile = gp.mask_raster_data(data, src.profile)

# rioxarray workflow
da = xr.open_rasterio("data.tif")
masked_da = gp.mask_raster_data(da)
```

### 5. Cloud Masking
#### `mask_clouds_S2`
**Description**: Masks clouds and optionally shadows in a Sentinel-2 raster image using various methods.

**Parameters**:
- `image_path` *(str)*: Path to the input raster image.
- `output_path` *(str, optional)*: Path to save the masked output raster. Defaults to the same directory as the input with '_masked' appended to the filename.
- `method` *(str, optional)*: The method for masking. Options are:
  - `'auto'`: Automatically chooses the best available method.
  - `'qa'`: Uses the QA60 band to mask clouds. WARNING: QA60 is masked between 2022-01-25 and 2024-02-28. Results for images in that date range could be wrong
  - `'probability'`: Uses the cloud probability band MSK_CLDPRB with a threshold for masking.
  - `'omnicloudmask'`: Utilizes OmniCloudMask for AI-based cloud detection. Might take a long time for big images
  - `'scl'`: Leverages the Scene Classification Layer (SCL) for masking.
  - `'standard'`: Similar to 'auto', but avoids the OmniCloudMask method.
- `mask_shadows` *(bool)*: Whether to mask cloud shadows. Defaults to `False`.
- `threshold` *(int, optional)*: Cloud probability threshold (if using a cloud probability band), from 0 to 100. Defaults to `20`.
- `qa60_idx` *(int, optional)*: Index of the QA60 band (1-based). Auto-detected if not provided.
- `qa60_path` *(str, optional)*: Path to the QA60 band (if in a separate file).
- `prob_band_idx` *(int, optional)*: Index of the cloud probability band (1-based). Auto-detected if not provided.
- `prob_band_path` *(str, optional)*: Path to the cloud probability band (if in a separate file).
- `scl_idx` *(int, optional)*: Index of the SCL band (1-based). Auto-detected if not provided.
- `scl_path` *(str, optional)*: Path to the SCL band (if in a separate file).
- `red_idx`, `green_idx`, `nir_idx` *(int, optional)*: Indices of the red, green, and NIR bands, respectively. Auto-detected if not provided.
- `nodata_value` *(float)*: Value for no-data regions. Defaults to `np.nan`.

**Returns**:
- *(str)*: The path to the saved masked output raster.

#### Example:
```python
import geopre as gp

output_s2 = gp.mask_clouds_S2("sentinel2_image.tif", method='auto', mask_shadows=True)
```

#### `mask_clouds_landsat`

**Description**:  
Masks clouds and optionally shadows in a Landsat raster image using various methods.

**Parameters**:

- **`image_path`** *(str)*: Path to the input multi-band raster image.  
- **`output_path`** *(str, optional)*: Path to save the masked output raster. Defaults to the same directory as the input with `_masked` suffix.  
- **`method`** *(str)*: The method for masking. Options are:  
  - **`'auto'`**: Automatically chooses the best available method.  
  - **`'qa'`**: Uses the QA_PIXEL band to mask clouds.  
  - **`'omnicloudmask'`**: Utilizes OmniCloudMask for AI-based cloud detection.  
- **`mask_shadows`** *(bool)*: Whether to mask cloud shadows. Defaults to `False`.  
- **`qa_pixel_path`** *(str, optional)*: Path to the separate QA_PIXEL raster file.  
- **`qa_pixel_idx`** *(int, optional)*: Index of the QA_PIXEL band (1-based).  
- **`confidence_threshold`** *(str, optional)*: Confidence threshold for cloud masking (e.g., `'Low'`, `'Medium'`, `'High'`). Defaults to `'High'`. WARNING: as per the Landsat official documentation, the confidence bands are still under development, always use the default 'High' untill further notice. [Source](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v6.pdf)
- **`red_idx`**, **`green_idx`**, **`nir_idx`** *(int, optional)*: Indices of the red, green, and NIR bands, respectively. Auto-detected if not provided.  
- **`nodata_value`** *(float)*: Value for no-data regions. Defaults to `np.nan`.  

### Returns

- *(str)*: The path to the saved masked output raster.  

### Example

```python
import geopre as gp

output_landsat = gp.mask_clouds_landsat("landsat_image.tif", method='auto', mask_shadows=True)
```

## 6. Water Masking

### `mask_water_S2`

**Description**:  
Masks water areas in Sentinel-2 imagery using the NDWI (Normalized Difference Water Index). Automatically detects Green and NIR bands based on band descriptions (e.g., B3, B8).

**Parameters**:
- `image_path` *(str)*: Path to the input raster image.
- `output_path` *(str, optional)*: Output path. If not specified, adds `_water_masked` suffix.
- `ndwi_threshold` *(float, optional)*: Threshold for NDWI. Default is 0.01.
- `nodata_value` *(float, optional)*: Value for masked (non-water) pixels. Default is `np.nan`.
- `green_idx`, `nir_idx` *(int, optional)*: Index of Green/NIR bands (1-based). Auto-detected if not provided.
- `green_path`, `nir_path` *(str, optional)*: If bands are stored in separate files.

**Returns**:
- *(str)*: Path to the saved water-masked output raster.

#### Example:
```python
import geopre as gp

masked_s2 = gp.mask_water_S2("sentinel_image.tif", ndwi_threshold=0.05)
```

### `mask_water_landsat`

**Description**:  
Same functionality as `mask_water_S2`, adapted for Landsat band naming (e.g., B3, B5, SR_B3, SR_B5).

**Parameters**:
- `image_path` *(str)*: Path to the input raster image.
- `output_path` *(str, optional)*: Output path. If not specified, adds `_water_masked` suffix.
- `ndwi_threshold` *(float, optional)*: Threshold for NDWI. Default is 0.01.
- `nodata_value` *(float, optional)*: Value for masked (non-water) pixels. Default is `np.nan`.
- `green_idx`, `nir_idx` *(int, optional)*: Index of Green/NIR bands (1-based). Auto-detected if not provided.
- `green_path`, `nir_path` *(str, optional)*: If bands are stored in separate files.

**Returns**:
- *(str)*: Path to the saved water-masked output raster.

#### Example:
```python
import geopre as gp

masked_landsat = gp.mask_water_landsat("landsat_image.tif", ndwi_threshold=0.05)
```

## 7. Band Stacking

### `stack_bands`

**Description**:  
Stacks multiple raster bands from a folder into a single multi-band raster. Support also .SAFE folders.

### Parameters

- **`input_path`** *(str or Path)*: Path to the folder containing band files.  
- **`required_bands`** *(list of str)*: List of band name identifiers (e.g., `["B4", "B3", "B2"]`).  
- **`output_path`** *(str or Path, optional)*: Path to save the stacked raster. Defaults to `"stacked.tif"` in the input folder.  
- **`resolution`** *(float, optional)*: Target resolution for resampling. Defaults to the highest available resolution.  

### Returns

- *(str)*: The path to the saved stacked output raster.  

### Example

```python
import geopre as gp

stacked_image = gp.stack_bands("/path/to/folder/containing/bands", ["B4", "B3", "B2"])
```
## Examples

We provide two example Jupyter notebooks demonstrating the usage of **GeoPre**:

- **[example_usage.ipynb](Example_Usage/example_usage.ipynb)** – Demonstrates **scaling, reprojecting, and masking operations**.
- **[example_usage_2.ipynb](Example_Usage/example_usage_2.ipynb)** – Covers **cloud masking, water masking and band stacking**.

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

## Changelog
**See the full release notes in the [CHANGELOG.md](CHANGELOG.md).**  

## License
This project is licensed under the MIT License. See LICENSE for more information.


## Author
Liang Zhongyou – [GitHub Profile](https://github.com/zyl009)

Matteo Gobbi Frattini – [GitHub Profile](https://github.com/MatteoGobbiF)

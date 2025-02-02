import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pyproj import CRS
import xarray as xr
"""
Geospatial data processing utilities for scaling, CRS handling, reprojection, and masking.

This module provides functions to handle common geospatial data operations:
- Normalization/scaling of raster data
- CRS management for vector and raster datasets
- Coordinate system reprojection
- No-data value masking

Functions handle multiple geospatial data types including:
- NumPy arrays
- GeoPandas GeoDataFrames
- Rasterio DatasetReaders
- Xarray DataArrays (rioxarray)
"""
#Standardization (Z-Score Scaling)
"""
This method centers the data around zero by subtracting the mean and dividing by the standard deviation.
Usage: Useful for machine learning models sensitive to outliers.
Example: Standardize a band of pixel values for clustering/classification.
"""
def Z_score_scaling(data):
    scaled_data = (data - np.mean(data)) / np.std(data)
    return scaled_data

#Min_Max_Scaling
"""
This method scales the pixel values to a fixed range, typically [0, 1] or [-1, 1].
Usage: Ideal when you want to preserve the relative range of values.
Example:For GeoTIFF image values (e.g., 0 to 65535), scale them to [0, 1].
"""
def Min_Max_Scaling(data):
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return scaled_data



#function to get crs of data
def get_crs(data):
    """
    Retrieve CRS from geospatial data objects.
    
    Handles:
    - GeoPandas GeoDataFrames (vector)
    - Rasterio DatasetReaders (raster)
    - Xarray DataArrays with rio accessor (raster)
    
    Args:
        data: Geospatial data object
        
    Returns:
        pyproj.CRS: Coordinate reference system or None if undefined
        
    Raises:
        ValueError: For unsupported input types
    """
    # Check for vector data (GeoDataFrame)
    if isinstance(data, gpd.GeoDataFrame):
        if data.crs:
            return data.crs
        else:
            return None
    
    # Check for rasterio Dataset (raster data)
    elif isinstance(data, rasterio.io.DatasetReader):
        if data.crs:
            return CRS.from_wkt(data.crs.to_wkt())  # Convert rasterio CRS to pyproj.CRS
        else:
            return None
    
    # Check for xarray DataArray with rioxarray accessor (raster data)
    elif hasattr(data, 'rio') and hasattr(data.rio, 'crs'):
        return data.rio.crs
    
    else:
        raise ValueError(
            "Unsupported data type. Expected one of: "
            "GeoDataFrame (vector), rasterio Dataset, or xarray DataArray with rio accessor (raster)."
        )


#function to compare crs of vector and raster data
def compare_crs(raster_obj, vector_gdf):
    """
    Compare CRS between raster and vector datasets.
    
    Args:
        raster_obj (DatasetReader/xarray.DataArray): Raster data source
        vector_gdf (gpd.GeoDataFrame): Vector data source
        
    Returns:
        dict: Comparison results with keys:
            - raster_crs: Formatted CRS string
            - vector_crs: Formatted CRS string  
            - same_crs: Boolean comparison result
            - error: Exception message if any
            
    Example:
        >>> compare_crs(raster, gdf)["same_crs"]
        True
    """
    result = {
        "raster_crs": None,
        "vector_crs": None,
        "same_crs": False,
        "error": None
    }

    try:
        # Handle different raster types
        if hasattr(raster_obj, 'rio'):  # rioxarray DataArray
            raster_crs = raster_obj.rio.crs
        elif hasattr(raster_obj, 'crs'):  # rasterio DatasetReader
            raster_crs = get_crs(raster_obj)  # Use get_crs for raster as well
        else:
            raise AttributeError("Unsupported raster type - use rioxarray.DataArray or rasterio.DatasetReader")
            
        # Handle vector data: Use get_crs to ensure correct retrieval
        vector_crs = get_crs(vector_gdf)

    except Exception as e:
        result["error"] = str(e)
        return result

    # Format CRS information
    def _format_crs(crs):
        if crs is None:
            return None
        try:
            return f"EPSG:{crs.to_epsg()}" if crs.to_epsg() else crs.to_wkt()
        except Exception:
            return str(crs)

    result["raster_crs"] = _format_crs(raster_crs)
    result["vector_crs"] = _format_crs(vector_crs)

    # Compare CRS
    try:
        if raster_crs is None and vector_crs is None:
            result["same_crs"] = False
        elif raster_crs is None or vector_crs is None:
            result["same_crs"] = False
        else:
            result["same_crs"] = (raster_crs == vector_crs)
    except Exception as e:
        result["error"] = str(e)

    return result


#function to reproject data
def reproject_data(data, target_crs):
    """
    Reproject geospatial data to target CRS.
    
    Supported inputs:
    - GeoDataFrames (vector reprojection)
    - Rasterio datasets (returns array + metadata)
    - Xarray objects (rioxarray reprojection)
    
    Args:
        data: Geospatial data object
        target_crs: CRS to reproject to (EPSG code/WKT/proj4 string)
        
    Returns:
        Reprojected data in format matching input type
        
    Raises:
        ValueError: If input lacks CRS definition
        TypeError: For unsupported input types
    """
    target_crs = CRS.from_user_input(target_crs)
    
    # Vector Data (GeoPandas)
    if isinstance(data, gpd.GeoDataFrame):
        if data.crs is None:
            raise ValueError("Vector data has no CRS. Cannot reproject.")
        if CRS(data.crs) == target_crs:
            return data
        return data.to_crs(target_crs)
    
    # Rasterio DatasetReader
    elif isinstance(data, rasterio.io.DatasetReader):
        src = data
        if src.crs is None:
            raise ValueError("Raster data has no CRS. Cannot reproject.")
        
        if CRS(src.crs) == target_crs:
            return src
        
        # Rasterio reprojection logic
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        
        dst_array = np.zeros((src.count, height, width), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, range(1, src.count + 1)),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )
        
        # Return reprojected array and new metadata
        return dst_array, {
            'driver': 'GTiff',
            'dtype': dst_array.dtype,
            'count': dst_array.shape[0],
            'width': width,
            'height': height,
            'transform': transform,
            'crs': target_crs
        }
    
    # rioxarray/xarray objects
    elif isinstance(data, (xr.DataArray, xr.Dataset)):
        if not data.rio.crs:
            raise ValueError("Raster data has no CRS")
        if CRS(data.rio.crs) == target_crs:
            return data
        return data.rio.reproject(target_crs.to_wkt())
    
    else:
        raise TypeError("Unsupported input type. Use: GeoDataFrame, xarray object, or rasterio DatasetReader")

#function to mask no value data
def mask_raster_data(data, profile=None, no_data_value=None, return_mask=False):
    """
    Mask no-data values in raster datasets.
    
    Handles both rasterio (numpy) and rioxarray (xarray) workflows.
    
    Args:
        data: Raster data (numpy.ndarray or xarray.DataArray)
        profile: Rasterio metadata dict (required for numpy arrays)
        no_data_value: Override for metadata's nodata value
        return_mask: Whether to return boolean mask
        
    Returns:
        Masked data array. For numpy inputs, returns tuple:
        (masked_array, profile). For xarray, returns DataArray.
        
    Raises:
        ValueError: If nodata value cannot be determined
        TypeError: For unsupported input types
    """
    # Handle xarray.DataArray (rioxarray)
    if isinstance(data, xr.DataArray):
        # Get no-data value from rioxarray metadata if not provided
        if no_data_value is None:
            no_data_value = data.rio.nodata
            if no_data_value is None:
                raise ValueError("No-data value not found in DataArray metadata. Specify `no_data_value`.")
        
        # Mask no-data values (replace them with NaN)
        masked_data = data.where(data != no_data_value)
        
        # Handle NaN values (if no-data is NaN, like in float rasters)
        if np.isnan(no_data_value):
            masked_data = data.where(~np.isnan(data))
        
        if return_mask:
            mask = ~masked_data.isnull()
            return masked_data, mask
        else:
            return masked_data

    # Handle NumPy array (rasterio)
    elif isinstance(data, np.ndarray):
        # Determine no-data value
        if no_data_value is None:
            if profile is not None:
                no_data_value = profile.get('nodata')
            else:
                raise ValueError("Specify `no_data_value` or provide a `profile` with `nodata`.")
        
        # Check if no_data_value is still None after retrieval
        if no_data_value is None:
            raise ValueError("Specify `no_data_value` or provide a `profile` with `nodata`.")
        
        # Handle NaN values (common in float rasters)
        if np.isnan(no_data_value):
            mask = ~np.isnan(data)
        else:
            mask = data != no_data_value
        
        # Create masked array
        masked_data = np.ma.masked_array(data, mask=~mask)
        
        if return_mask:
            return masked_data, mask, profile
        else:
            return masked_data, profile
    
    else:
        raise TypeError("Unsupported data type. Input must be `numpy.ndarray` or `xarray.DataArray`.")
import numpy as np
import numpy.ma as ma
import geopandas as gpd
import rasterio
from pyproj import CRS
import rioxarray as rxr
import xarray as xr

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
    Retrieve the Coordinate Reference System (CRS) from an in-memory geospatial object (vector or raster).

    Parameters:
    data: Input geospatial data. Supported types:
        - Vector: `geopandas.GeoDataFrame`
        - Raster: `rasterio.io.DatasetReader` (opened raster dataset) or `xarray.DataArray` (with `rio` accessor).

    Returns:
    pyproj.CRS: The CRS of the geospatial data. Returns `None` if the CRS is not defined.

    Raises:
    ValueError: If the input data type is unsupported.
    """
    # Check for vector data (GeoDataFrame)
    if isinstance(data, gpd.GeoDataFrame):
        return data.crs
    
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
    Compare CRS of raster (xarray.DataArray or rasterio) and vector (geopandas) data.
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
            raster_crs = CRS.from_wkt(raster_obj.crs.wkt)
        else:
            raise AttributeError("Unsupported raster type - use rioxarray.DataArray or rasterio.DatasetReader")
            
        # Handle vector data
        vector_crs = vector_gdf.crs

    except Exception as e:
        result["error"] = str(e)
        return result

    # Format CRS information
    def _format_crs(crs):
        if crs is None:
            return "No CRS defined"
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
            result["same_crs"] = (raster_crs==vector_crs)
    except Exception as e:
        result["error"] = str(e)

    return result


#function to reproject data
def reproject_data(data, target_crs):
    """
    Modern CRS handling without PROJ4 strings
    """
    # Convert input to CRS object using pyproj's auto-detection
    target_crs = CRS.from_user_input(target_crs)
    
    # Vector Data (GeoPandas)
    if isinstance(data, gpd.GeoDataFrame):
        if data.crs is None:
            raise ValueError("Vector data has no CRS. Cannot reproject.")
        
        # Compare CRS objects directly
        if data.crs == target_crs:
            return data
            
        # Use CRS object directly
        return data.to_crs(target_crs)
    
    # Raster Data (xarray with rioxarray)
    elif isinstance(data, (xr.DataArray, xr.Dataset)):
        if not data.rio.crs:
            raise ValueError("Raster data has no CRS. Cannot reproject.")
            
        # Compare using pyproj.CRS objects
        if CRS(data.rio.crs) == target_crs:
            return data
            
        # Reproject 
        return data.rio.reproject(target_crs.to_wkt())
    
    else:
        raise TypeError("Supported types: GeoDataFrame, xarray DataArray/Dataset")

#function to mask no value data
def mask_raster_data(data, profile=None, no_data_value=None, return_mask=False):
    """
    Mask no-data values in raster data loaded with `rasterio` or `rioxarray`.
    
    Args:
        data (numpy.ndarray or xarray.DataArray): Raster data.
        profile (dict, optional): Raster metadata from `rasterio` (used if `data` is a NumPy array).
        no_data_value (int/float, optional): Explicit no-data value (overrides metadata).
        return_mask (bool): If True, returns the boolean mask. Default: False.
        
    Returns:
        masked_data: Masked array (NumPy `MaskedArray` or `xarray.DataArray`) with no-data values masked.
        mask (optional): Boolean mask (True = valid data). Returned if `return_mask=True`.
        profile (optional): Original metadata (if provided and input is a NumPy array).
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







#example usage
# Load vector data
vector = gpd.read_file("/mnt/CA6A062B6A06153B/study/Geospatial processing/test/Municipalities_on_lakes.shp")
# Open raster
raster = rxr.open_rasterio("/mnt/CA6A062B6A06153B/study/Geospatial processing/test/l8monzaali.tif")# ,masked=True)

#get crs
raster_crs=get_crs(raster)
vector_crs=get_crs(vector)
print(raster_crs)
print(vector_crs)

#compare crs
result = compare_crs(raster, vector)
print(result)

# Reproject
reprojected_vector = reproject_data(vector, "EPSG:4326")
reprojected_raster = reproject_data(raster, "EPSG:3857")

# Save output (maintains all geospatial information)
reprojected_raster.rio.to_raster("/mnt/CA6A062B6A06153B/study/Geospatial processing/test/output.tif")

print(get_crs(reprojected_raster))
print(get_crs(reprojected_vector))




#mask no value data
#for raster loded by rioxarray
data = rxr.open_rasterio("/mnt/CA6A062B6A06153B/study/Geospatial processing/test/output.tif")
masked_data, mask = mask_raster_data(data, return_mask=True)
print("masked_data:",masked_data)
print("mask:",mask)

"""
#Undefined No-Data: If the raster lacks nodata metadata, specify it manually:
# Specify a no-data value explicitly (e.g., -9999)
no_data_value = -9999
masked_data, mask = mask_raster_data(data, no_data_value=no_data_value, return_mask=True)
"""

#Multi-Band Masks: To mask pixels where any band has no-data:
combined_mask = mask.all(dim="band")  # Mask where all bands are valid
masked_data = data.where(combined_mask)
# Flatten the data for scikit-learn
valid_data = masked_data.stack(samples=("y", "x")).dropna(dim="samples")
print("valid_data:",valid_data)

#for raster loaded by rasterio
with rasterio.open("/mnt/CA6A062B6A06153B/study/Geospatial processing/test/output.tif") as src:
    data = src.read()
    profile = src.profile
masked_data, mask, profile = mask_raster_data(data, profile=profile, return_mask=True)
print("maksed_data:",masked_data)
print("mask:",mask)
"""
#If the raster lacks nodata metadata,specify a no-data value explicitly (e.g., -9999)
no_data_value = -9999
# Mask the raster data
masked_data, mask, profile = mask_raster_data(data, no_data_value=no_data_value, return_mask=True)
"""
# Create a combined mask for all bands (valid if all bands are valid)
multi_band_mask = np.all(mask, axis=0)  # Shape: [height, width]
masked_data = ma.masked_array(data, mask=np.broadcast_to(~mask, data.shape))

# For classification, extract valid pixels (across all bands)
valid_samples = masked_data.compressed()  # Flattened array of valid values
print("valid_samples:",valid_samples)



#z_ score scaling
raster=Z_score_scaling(raster)


#min_max scaling
raster=Min_Max_Scaling(raster)

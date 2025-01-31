import numpy as np
import geopandas as gpd
import rasterio
from pyproj import CRS
import rioxarray as rxr
import xarray as xr

#Standardization (Z-Score Scaling)
#Usage: Useful for machine learning models sensitive to outliers.
#Example: Standardize a band of pixel values for clustering/classification.
def scaled_data(data):
    scaled_data = (data - np.mean(data)) / np.std(data)
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

raster=scaled_data(raster)



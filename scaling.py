import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS

#Standardization (Z-Score Scaling)
#Usage: Useful for machine learning models sensitive to outliers.
#Example: Standardize a band of pixel values for clustering/classification.
data=gpd.read_file("input.geojson")


scaled_data = (data - np.mean(data)) / np.std(data)

#function to reproject crs
def reproject_data(data, target_crs):
    """
    Reproject geospatial data to a target CRS if not already in it.
    
    Args:
        data: A GeoDataFrame (vector) or rasterio DatasetReader (raster).
        target_crs: Target CRS (accepts EPSG codes, proj4 strings, etc.).
        
    Returns:
        Reprojected data in the same format as input.
        
    Raises:
        TypeError: If input data type is unsupported.
        ValueError: If source CRS is missing.
    """
    target_crs = CRS(target_crs)

    # Vector Data Handling (GeoPandas)
    if isinstance(data, gpd.GeoDataFrame):
        if data.crs is None:
            raise ValueError("Vector data has no CRS. Cannot reproject.")
        if CRS(data.crs) == target_crs:
            return data
        return data.to_crs(target_crs.to_dict())

    # Raster Data Handling (rasterio)
    elif isinstance(data, rasterio.io.DatasetReader):
        src_crs = data.crs
        if src_crs is None:
            raise ValueError("Raster data has no CRS. Cannot reproject.")
        if CRS(src_crs) == target_crs:
            return data

        # Calculate new transformation and dimensions
        transform, width, height = calculate_default_transform(
            src_crs, target_crs, data.width, data.height, *data.bounds
        )

        # Create destination array
        dst_array = np.zeros((data.count, height, width), dtype=data.dtypes[0])

        # Reproject each band
        for i in range(1, data.count + 1):
            reproject(
                source=rasterio.band(data, i),
                destination=dst_array[i-1],
                src_transform=data.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )

        # Update metadata
        dst_profile = data.profile.copy()
        dst_profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        return dst_array, dst_profile

    else:
        raise TypeError("Unsupported data type. Use GeoDataFrame or rasterio DatasetReader.")



#function to compare crs of vector and raster data
def compare_crs(raster_obj, vector_gdf):
    """
    Compare the CRS of pre-loaded raster and vector data.

    Parameters:
    raster_obj (rasterio.DatasetReader): Loaded raster object from rasterio.
    vector_gdf (geopandas.GeoDataFrame): Loaded vector data from geopandas.

    Returns:
    dict: Contains CRS info for raster/vector and a boolean indicating if they match.
    """
    result = {
        "raster_crs": None,
        "vector_crs": None,
        "same_crs": False,
        "error": None
    }

    # Get raster CRS
    try:
        raster_crs = raster_obj.crs
        if raster_crs is not None:
            # Convert rasterio CRS to pyproj.CRS for comparison
            raster_crs = CRS.from_wkt(raster_obj.crs.wkt)
    except Exception as e:
        result["error"] = f"Raster CRS error: {e}"
        return result

    # Get vector CRS
    try:
        vector_crs = vector_gdf.crs
    except Exception as e:
        result["error"] = f"Vector CRS error: {e}"
        return result

    # Generate user-friendly CRS representations
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
            result["same_crs"] = True
        elif raster_crs is None or vector_crs is None:
            result["same_crs"] = False
        else:
            result["same_crs"] = raster_crs.equals(vector_crs)
    except Exception as e:
        result["error"] = f"Comparison error: {e}"

    return result
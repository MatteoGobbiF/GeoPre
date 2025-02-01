from scaling_and_reproject import Z_score_scaling, Min_Max_Scaling,get_crs, compare_crs, reproject_data,mask_raster_data
import geopandas as gpd
import rasterio
from pyproj import CRS
import pytest
import numpy as np
from rasterio.transform import from_origin
import rioxarray as rxr

@pytest.fixture 
def create_data():
    data = np.array([[10, 20, 30], [40, 50, 60]])
    return data

def test_z_score_scaling():
    scaled_data = Z_score_scaling(create_data)
    assert np.allclose(np.mean(scaled_data), 0, atol=1e-6)  # Mean should be ~0
    assert np.allclose(np.std(scaled_data), 1, atol=1e-6)   # Std should be ~1

def test_min_max_scaling():
    scaled_data = Min_Max_Scaling(create_data)
    assert scaled_data.min() == 0  # Min should be 0
    assert scaled_data.max() == 1  # Max should be 1

@pytest.fixture
def create_vector_with_crs():
    """Create a dummy GeoDataFrame with a CRS."""
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

@pytest.fixture
def create_vector_without_crs():
    """Create a dummy GeoDataFrame with a CRS."""
    return gpd.GeoDataFrame(geometry=[])

@pytest.fixture
def create_tif_with_crs(tmp_path):
    """Create a dummy GeoTIFF with CRS for testing."""
    file_path = tmp_path / "test_with_crs.tif"
    data = np.random.rand(50, 50).astype(np.float32)
    transform = from_origin(0, 0, 1, 1)

    meta = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype,
        "transform": transform,
        "crs": "EPSG:4326",
        "nodata": -9999
    }

    with rasterio.open(file_path, "w", **meta) as dst:
        dst.write(data, 1)

    return file_path

@pytest.fixture
def create_tif_without_crs(tmp_path):
    """Create a dummy GeoTIFF with no CRS for testing."""
    file_path = tmp_path / "test_no_crs.tif"
    data = np.random.rand(50, 50).astype(np.float32)
    transform = from_origin(0, 0, 1, 1)

    meta = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype,
        "transform": transform,
        "nodata": -9999
    }

    with rasterio.open(file_path, "w", **meta) as dst:
        dst.write(data, 1)

    return file_path


def test_get_crs_vector_with_crs():
    assert get_crs(create_vector_with_crs) == CRS("EPSG:4326")

def test_get_crs_vector_without_crs():
    assert get_crs(create_vector_without_crs) == None

def test_get_crs_raster_with_crs(create_tif_with_crs):
    """Test get_crs() when raster has CRS."""
    with rasterio.open(create_tif_with_crs) as raster:
        assert get_crs(raster).to_epsg() == 4326  

def test_get_crs_raster_without_crs(create_tif_without_crs):
    """Test get_crs() when raster has CRS."""
    with rasterio.open(create_tif_without_crs) as raster:
        assert get_crs(raster).to_epsg() == None


def test_reproject_vector_with_crs():
    reprojected_gdf = reproject_data(create_vector_with_crs,"EPSG:3857")
    assert reprojected_gdf.crs == CRS("EPSG:3857")

def test_reproject_raster_with_crs():
    raster = rxr.open_rasteri(create_tif_with_crs)
    reproject_raster=reproject_data(raster, "EPSG:3857")
    assert reproject_raster.rio.crs.to_epsg()== 3857

def test_reproject_vector_without_crs():
    reprojected_gdf = reproject_data(create_vector_without_crs,"EPSG:3857")
    with pytest.raises(ValueError, match="Vector data has no CRS. Cannot reproject."):
            reproject_data(reprojected_gdf, "EPSG:4326")

def test_reproject_raster_without_crs(create_tif_without_crs):
    with rasterio.open(create_tif_without_crs) as raster:
        with pytest.raises(ValueError, match="Raster data has no CRS. Cannot reproject."):
            reproject_data(raster, "EPSG:4326")

def test_compare_crs_two_vector():
    gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf2 = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    assert not compare_crs(gdf, gdf2)["same_crs"]

def test_compare_crs_raster__without_crs(create_tif_without_crs, create_vector_with_crs):
    """Test compare_crs() when raster has no CRS."""
    with rasterio.open(create_tif_without_crs) as raster:
        vector = create_vector_with_crs
        result = compare_crs(raster, vector)

        assert result["raster_crs"] == "No CRS defined"
        assert result["vector_crs"] == "EPSG:4326"
        assert result["same_crs"] is False
        assert result["error"] is None


def test_compare_crs_raster_with_crs(create_tif_with_crs, create_vector_with_crs):
    """Test compare_crs() when both raster and vector have CRS."""
    with rasterio.open(create_tif_with_crs) as raster:
        vector = create_vector_with_crs
        result = compare_crs(raster, vector)

        assert result["same_crs"] is True




@pytest.fixture
def create_tif_with_nodata(tmp_path):
    """Create a dummy GeoTIFF with no-data values for testing."""
    file_path = tmp_path / "test_mask.tif"
    data = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, -9999, 9, 10],
        [11, 12, 13, -9999, 15],
        [16, -9999, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ]).astype(np.float32)

    meta = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype,
        "transform": from_origin(0, 0, 1, 1),
        "crs": "EPSG:4326",
        "nodata": -9999
    }

    with rasterio.open(file_path, "w", **meta) as dst:
        dst.write(data, 1)

    return file_path

def test_mask_raster_data(create_tif_with_nodata):
    """Test mask_raster_data() correctly masks no-data values."""
    with rasterio.open(create_tif_with_nodata) as raster:
        data = raster.read(1)
        profile = raster.profile

    masked_data, _ = mask_raster_data(data, profile)

    assert np.ma.is_masked(masked_data), "Expected masked array but got unmasked data."
    assert np.ma.is_masked(masked_data[1, 2]), "Expected (1,2) to be masked."
    assert np.ma.is_masked(masked_data[2, 3]), "Expected (2,3) to be masked."

def test_mask_raster_data_no_nodata(create_tif_with_nodata):
    """Test mask_raster_data() raises an error if no-data value is not provided."""
    with rasterio.open(create_tif_with_nodata) as raster:
        data = raster.read(1)
        profile = raster.profile

    profile.pop("nodata")  # Remove nodata from metadata

    with pytest.raises(ValueError, match="Specify `no_data_value` or provide a `profile` with `nodata`."):
        mask_raster_data(data, profile)

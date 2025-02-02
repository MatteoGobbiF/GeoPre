import numpy as np
import rasterio as rio
from pathlib import Path
from rasterio.profiles import DefaultGTiffProfile
from typing import Union, List

def stack_bands(input_path: Union[Path, str], required_bands: List[str], output_path: Union[Path, str] = None, resolution: float = None) -> str:
    """
    Stacks multiple raster bands into a single multi-band raster.

    Parameters:
        input_path (str or Path): Path to the folder containing band files.
        required_bands (list of str): List of band name identifiers (e.g., ["B4", "B3", "B2"]).
        output_path (str or Path, optional): Path to save the stacked raster. If not provided, it is saved in the same directory as `input_path` with the name "stacked.tif".
        resolution (float, optional): Target resolution for resampling. If None, the highest resolution available is used.

    Returns:
        str: The path to the saved stacked output raster.
    """
    input_path = Path(input_path)

    # Set default output path if not provided
    if output_path is None:
        output_path = input_path / "stacked.tif"

    # Dictionary to store found band files
    band_files = {}
    found_bands = []
    dtypes = set()
    
    # Iterate through the required bands and search for corresponding files
    for band_name in required_bands:
        try:
            # Look for files matching the band name (TIF or JP2 format)
            band_files[band_name] = list(input_path.rglob(f"*{band_name}*.TIF")) or list(input_path.rglob(f"*{band_name}*.jp2"))
            
            # If a file is found, store the first match
            if band_files[band_name]:
                band_files[band_name] = band_files[band_name][0]
            else:
                raise IndexError  # Raise an exception if no file is found
        except IndexError:
            print(f"Warning: Band {band_name} not found in {input_path}, skipping.")
    
    # Ensure at least one valid band was found
    if not band_files:
        raise ValueError("No valid bands found. Check your file names and folder.")
    
    data = []  # List to hold band data
    profile = None  # Metadata profile of the raster
    crs_set = set()  # Set to store unique coordinate reference systems (CRS)
    resolutions = set()  # Set to store unique resolutions
    
    # Read each band file and process the data
    for band_name, band_path in band_files.items():
        with rio.open(band_path) as src:
            if profile is None:
                profile = src.profile  # Store the first raster's metadata
            
            crs_set.add(src.crs)  # Store CRS
            resolutions.add((src.res[0], src.res[1]))  # Store resolution
            dtypes.add(src.dtypes[0])  # Store data type
            
            # Determine the native resolution of the raster
            native_resolution = int(src.res[0])
            
            # If no resolution is specified, use the highest resolution available
            if resolution is None:
                resolution = max(res[0] for res in resolutions)
            
            # Compute resampling scale factor
            scale_factor = native_resolution / resolution
            
            # Handle multi-band files correctly
            if src.count > 1:
                # Retrieve band descriptions or generate default names
                existing_band_descriptions = src.descriptions if any(src.descriptions) else [f"{band_name}_B{i}" for i in range(1, src.count + 1)]
                
                for i in range(1, src.count + 1):  # Read all bands
                    if native_resolution == resolution:
                        data.append(src.read(i))
                    else:
                        # Resample band to target resolution
                        data.append(
                            src.read(
                                i,
                                out_shape=(
                                    int(src.height * scale_factor),
                                    int(src.width * scale_factor),
                                ),
                            )
                        )
                    found_bands.append(existing_band_descriptions[i-1])  # Store band names
            else:
                if native_resolution == resolution:
                    data.append(src.read(1))
                else:
                    # Resample band to target resolution
                    data.append(
                        src.read(
                            1,
                            out_shape=(
                                int(src.height * scale_factor),
                                int(src.width * scale_factor),
                            ),
                        )
                    )
                found_bands.append(band_name)  # Store band name
    
    # Warnings for potential data inconsistencies
    if len(crs_set) > 1:
        print("Warning: Different CRS detected. Ensure compatibility before analysis.")
    if len(resolutions) > 1:
        print("Warning: Different resolutions detected. Consider resampling before stacking.")
    if len(dtypes) > 1:
        print("Warning: Different data types detected among bands. The output type may be automatically adjusted.")
    
    # Convert data to a NumPy array for easier manipulation
    data = np.array(data)
    
    # Update the transform (spatial reference and resolution)
    profile["transform"] = rio.transform.from_origin(
        profile["transform"][2],
        profile["transform"][5],
        resolution,
        resolution,
    )
    
    # Update profile metadata for output raster
    profile.update(
        count=len(data),  # Number of bands
        dtype=data.dtype,  # Data type
        height=data.shape[1],  # Image height
        width=data.shape[2],  # Image width
        driver="GTiff",  # Output format (GeoTIFF)
    )
    
    # Write the stacked raster to the output file
    with rio.open(output_path, "w", **profile) as dst:
        for i in range(data.shape[0]):  # Iterate through bands
            dst.write(data[i], i + 1)  # Write each band separately
        
        # Set band descriptions for better readability
        dst.descriptions = tuple(found_bands)
    
    print(f"Stacked raster saved at {output_path}")

    return str(output_path)  # Return the path of the saved file
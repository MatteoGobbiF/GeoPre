import rasterio as rio
import os
import numpy as np

def mask_water_S2(image_path, output_path=None, ndwi_threshold=0.01, nodata_value=np.nan, 
                  green_idx=None, green_path=None, nir_idx=None, nir_path=None):
    """
    Masks water areas in a Sentinel-2 raster image using the NDWI index.

    Parameters:
        image_path (str): Path to the input raster image.
        output_path (str, optional): Path to save the masked output raster. 
                                     If not provided, the output will be saved with '_water_masked' suffix.
        ndwi_threshold (float, optional): Threshold for NDWI to identify water. Defaults to 0.01.
        nodata_value (float, optional): Value to assign to masked (non-water) pixels. Defaults to np.nan.
        green_idx (int, optional): Index of the Green band (1-based). Auto-detected if not provided.
        green_path (str, optional): Path to the Green band (if in a separate file).
        nir_idx (int, optional): Index of the NIR band (1-based). Auto-detected if not provided.
        nir_path (str, optional): Path to the NIR band (if in a separate file).

    Returns:
        str: The path to the saved water-masked output raster.
    """
    try:
        # Validate NDWI threshold
        try:
            ndwi_threshold = float(ndwi_threshold)
            if not (-1.0 <= ndwi_threshold <= 1.0):
                raise ValueError("NDWI threshold should be between -1.0 and 1.0.")
        except Exception as e:
            raise ValueError(f"Invalid NDWI threshold value: {ndwi_threshold}. Must be a float between -1 and 1.") from e

        # Set default output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_water_masked{ext}"

        with rio.open(image_path) as src:
            metadata = src.meta.copy()
            band_descriptions = list(src.descriptions)

            # --- Green band loading ---
            green = None
            if green_path:
                print(f"Using external Green band file: {green_path}")
                with rio.open(green_path) as green_src:
                    green = green_src.read(1).astype(np.float32)
            else:
                if green_idx is not None:
                    print(f"Using Green band at index: {green_idx}")
                else:
                    if 'B3' in band_descriptions:
                        green_idx = band_descriptions.index('B3') + 1
                    elif 'B03' in band_descriptions:
                        green_idx = band_descriptions.index('B03') + 1
                    else:
                        raise ValueError("Green band (B3/B03) not found in input image.")
                    print(f"Auto-detected Green band at index: {green_idx}")
                green = src.read(green_idx).astype(np.float32)

            # --- NIR band loading ---
            nir = None
            if nir_path:
                print(f"Using external NIR band file: {nir_path}")
                with rio.open(nir_path) as nir_src:
                    nir = nir_src.read(1).astype(np.float32)
            else:
                if nir_idx is not None:
                    print(f"Using NIR band at index: {nir_idx}")
                else:
                    if 'B8' in band_descriptions:
                        nir_idx = band_descriptions.index('B8') + 1
                    elif 'B08' in band_descriptions:
                        nir_idx = band_descriptions.index('B08') + 1
                    elif 'B8A' in band_descriptions:
                        nir_idx = band_descriptions.index('B8A') + 1
                    else:
                        raise ValueError("NIR band (B8/B08/B8A) not found in input image.")
                    print(f"Auto-detected NIR band at index: {nir_idx}")
                nir = src.read(nir_idx).astype(np.float32)

            # Compute NDWI = (Green - NIR) / (Green + NIR)
            ndwi = (green - nir) / (green + nir + 1e-10)  # Avoid division by zero

            # Create mask for water pixels (NDWI >= threshold)
            water_mask = ndwi >= ndwi_threshold

            # Apply mask to all bands
            masked_data = []
            for band in range(1, src.count + 1):
                data = src.read(band).astype(np.float32)
                masked_band = np.where(water_mask, nodata_value, data)
                masked_data.append(masked_band)

            masked_data = np.array(masked_data)

            # Update metadata
            metadata.update({"nodata": nodata_value, "dtype": 'float32'})

            # Write output file
            with rio.open(output_path, "w", **metadata) as dest:
                dest.write(masked_data.astype('float32'))
                for i, desc in enumerate(band_descriptions):
                    if desc:
                        dest.set_band_description(i + 1, desc)

        print(f"Water-masked image saved to {output_path}.")
        return output_path

    except Exception as e:
        print(f"An error occurred during water masking: {e}")


def mask_water_landsat(image_path, output_path=None, ndwi_threshold=0.01, nodata_value=np.nan, 
                  green_idx=None, green_path=None, nir_idx=None, nir_path=None):
    """
    Masks water areas in a Landsat raster image using the NDWI index.

    Parameters:
        image_path (str): Path to the input raster image.
        output_path (str, optional): Path to save the masked output raster. 
                                     If not provided, the output will be saved with '_water_masked' suffix.
        ndwi_threshold (float, optional): Threshold for NDWI to identify water. Defaults to 0.01.
        nodata_value (float, optional): Value to assign to masked (non-water) pixels. Defaults to np.nan.
        green_idx (int, optional): Index of the Green band (1-based). Auto-detected if not provided.
        green_path (str, optional): Path to the Green band (if in a separate file).
        nir_idx (int, optional): Index of the NIR band (1-based). Auto-detected if not provided.
        nir_path (str, optional): Path to the NIR band (if in a separate file).

    Returns:
        str: The path to the saved water-masked output raster.
    """
    try:
        # Validate NDWI threshold
        try:
            ndwi_threshold = float(ndwi_threshold)
            if not (-1.0 <= ndwi_threshold <= 1.0):
                raise ValueError("NDWI threshold should be between -1.0 and 1.0.")
        except Exception as e:
            raise ValueError(f"Invalid NDWI threshold value: {ndwi_threshold}. Must be a float between -1 and 1.") from e

        # Set default output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_water_masked{ext}"

        with rio.open(image_path) as src:
            metadata = src.meta.copy()
            band_descriptions = list(src.descriptions)

            # --- Green band loading ---
            green = None
            if green_path:
                print(f"Using external Green band file: {green_path}")
                with rio.open(green_path) as green_src:
                    green = green_src.read(1).astype(np.float32)
            else:
                if green_idx is not None:
                    print(f"Using Green band at index: {green_idx}")
                else:
                    if 'B3' in band_descriptions:
                        green_idx = band_descriptions.index('B3') + 1
                    elif 'B03' in band_descriptions:
                        green_idx = band_descriptions.index('B03') + 1
                    elif 'SR_B3' in band_descriptions:
                        green_idx = band_descriptions.index('SR_B3') + 1
                    else:
                        raise ValueError("Green band (B3/B03/SR_B3) not found in input image.")
                    print(f"Auto-detected Green band at index: {green_idx}")
                green = src.read(green_idx).astype(np.float32)

            # --- NIR band loading ---
            nir = None
            if nir_path:
                print(f"Using external NIR band file: {nir_path}")
                with rio.open(nir_path) as nir_src:
                    nir = nir_src.read(1).astype(np.float32)
            else:
                if nir_idx is not None:
                    print(f"Using NIR band at index: {nir_idx}")
                else:
                    if 'B5' in band_descriptions:
                        nir_idx = band_descriptions.index('B5') + 1
                    elif 'B05' in band_descriptions:
                        nir_idx = band_descriptions.index('B05') + 1
                    elif "SR_B5" in band_descriptions:
                        nir_idx = band_descriptions.index("SR_B5") + 1
                    else:
                        raise ValueError("NIR band (B5/B05/SR_B5) not found in input image.")
                    print(f"Auto-detected NIR band at index: {nir_idx}")
                nir = src.read(nir_idx).astype(np.float32)

            # Compute NDWI = (Green - NIR) / (Green + NIR)
            ndwi = (green - nir) / (green + nir + 1e-10)  # Avoid division by zero

            # Create mask for water pixels (NDWI >= threshold)
            water_mask = ndwi >= ndwi_threshold

            # Apply mask to all bands
            masked_data = []
            for band in range(1, src.count + 1):
                data = src.read(band).astype(np.float32)
                masked_band = np.where(water_mask, nodata_value, data)
                masked_data.append(masked_band)

            masked_data = np.array(masked_data)

            # Update metadata
            metadata.update({"nodata": nodata_value, "dtype": 'float32'})

            # Write output file
            with rio.open(output_path, "w", **metadata) as dest:
                dest.write(masked_data.astype('float32'))
                for i, desc in enumerate(band_descriptions):
                    if desc:
                        dest.set_band_description(i + 1, desc)

        print(f"Water-masked image saved to {output_path}.")
        return output_path

    except Exception as e:
        print(f"An error occurred during water masking: {e}")
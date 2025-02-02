import rasterio
import numpy as np
import os

from omnicloudmask import predict_from_array

def omnicloudmask(red_band, green_band, nir_band, mask_shadows=False):
    """
    Applies an omni-cloud masking method using red, green, and NIR bands, using the omnicloudmask library.
    https://github.com/DPIRD-DMA/OmniCloudMask

    Parameters:
        red_band (numpy.ndarray): The red band of the image.
        green_band (numpy.ndarray): The green band of the image.
        nir_band (numpy.ndarray): The NIR band of the image. (As per the OmniCloudMask documentation, reasonable results also with blue band)
        mask_shadows (bool): Whether to mask shadows.

    Returns:
        numpy.ndarray: A mask where 1 indicates cloud or shadow (depending on the mask_shadows parameter) and 0 indicates clear.
    """
    # Stack bands into a single array with dimensions (height, width, channels)
    stacked_array = np.stack((red_band, green_band, nir_band), axis=0)
    # Use predict_from_array to generate the cloud mask
    print("This could take a few minutes")
    prediction = predict_from_array(stacked_array, no_data_value='nan')
    # Adjust mask based on the requirement to mask shadows
    if mask_shadows:
        # Mask everything that is not clear (0)
        cloud_mask = prediction != 0
    else:
        # Mask only thick (1) and thin clouds (2)
        cloud_mask = (prediction == 1) | (prediction == 2)
    
    cloud_mask = np.squeeze(cloud_mask)  

    return cloud_mask

def mask_clouds_S2(image_path, output_path=None, method='auto', mask_shadows=False, nodata_value=np.nan, threshold=20, 
                   qa60_idx=None, qa60_path=None, prob_band_idx=None, prob_band_path=None, scl_idx=None, scl_path=None, 
                   red_idx=None, green_idx=None, nir_idx=None):
    """
    Masks clouds and optionally shadows in a Sentinel-2 raster image using various methods.

    Parameters:
        image_path (str): Path to the input raster image.
        output_path (str, optional): Path to save the masked output raster. If not provided, the output will be saved to the same directory as the input with '_masked' appended to the filename.
        method (str, optional): The method for masking ('auto', 'qa', 'probability', 'omnicloudmask', 'scl', 'standard'). Defaults to 'auto'
        mask_shadows (bool): Whether to mask cloud shadows (applicable for SCL and omnicloudmask methods). Defaults to False
        threshold (int): Cloud probability threshold (if using cloud probability band), from 0 to 100. Defaults to 20 
        qa60_idx (int, optional): Index of the QA60 band (1-based, if specified manually). Auto-detected if not provided.
        qa60_path (str, optional): Path to the QA60 band (if in a separate file).
        prob_band_idx (int, optional): Index of the cloud probability band (1-based, if specified manually). Auto-detected if not provided.
        prob_band_path (str, optional): Path to the cloud probability band (if in a separate file).
        scl_idx (int, optional): Index of the SCL band (1-based, for classification masking). Auto-detected if not provided.
        scl_path (str, optional): Path to the SCL band (if in a separate file).
        red_idx (int, optional): Index of the red band (1-based, for omnicloudmask). Auto-detected if not provided.
        green_idx (int, optional): Index of the green band (1-based, for omnicloudmask). Auto-detected if not provided.
        nir_idx (int, optional): Index of the NIR band (1-based, for omnicloudmask). Auto-detected if not provided.
        
    Returns:
        str: The path to the saved masked output raster
    """
    try:
        # Set default output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_masked{ext}"

        # Open the raster file
        with rasterio.open(image_path) as src:
            # Read metadata
            metadata = src.meta.copy()
            band_descriptions = list(src.descriptions)

            cloud_mask = None

            # Handle auto mode and shadow masking logic
            if method == 'auto' and mask_shadows:
                print("mask_shadows=True: Skipping QA60 and MSK_CLDPRB, prioritizing SCL and omnicloudmask methods.")
                method = 'shadows'
            if method == 'standard' and mask_shadows:
                print("mask_shadows=True: Skipping QA60 and MSK_CLDPRB, prioritizing SCL method.")
                method = 'scl'

            if method in ('auto', 'probability', 'standard') and cloud_mask is None:
                print("Trying with Cloud Probability Band (MSK_CLDPRB)")
                cloud_prob = None

                # Check for MSK_CLDPRB band:                
                if prob_band_path: # Check if band provided in separate file
                    print(f"Using external cloud probability band file: {prob_band_path}")
                    with rasterio.open(prob_band_path) as prob_src:
                        cloud_prob = prob_src.read(1)
                else: # Check if band idx is provided, otherwise tries to automatically find the band
                    if prob_band_idx is None and 'MSK_CLDPRB' in src.descriptions:
                        prob_band_idx = src.descriptions.index('MSK_CLDPRB') + 1
                    if prob_band_idx:
                        cloud_prob = src.read(prob_band_idx)
                        print("Cloud probability band (MSK_CLDPRB) found at index" + str(prob_band_idx))
                

                if cloud_prob is not None:
                    if mask_shadows:
                        print("Warning: MSK_CLDPRB does not support shadow masking. Proceeding with cloud masking only.")
                    print("Cloud probability band (MSK_CLDPRB) found. Using it for cloud masking with a threshold of " + str(threshold))

                    # Create a cloud mask using the threshold
                    cloud_mask = cloud_prob >= threshold
                else:
                    print("MSK_CLDPRB band not found!")

            if method in ('auto', 'scl', 'standard', 'shadows') and cloud_mask is None:
                print("Trying with Scene Classification Layer (SCL)")
                scl = None
                # Check for SCL band
                if scl_path: # Check if band provided in separate file
                    print(f"Using external SCL band file: {scl_path}")
                    with rasterio.open(scl_path) as scl_src:
                        scl = scl_src.read(1)
                else: # Check if band idx is provided, otherwise tries to automatically find the band
                    if scl_idx is None and 'SCL' in src.descriptions:
                        scl_idx = src.descriptions.index('SCL') + 1
                    if scl_idx:
                        scl = src.read(scl_idx)
                        print("SCL band found at index" + str(scl_idx))

                if scl is not None:
                    print("SCL band found. Using it for classification masking.")

                    # Create masks for clouds and optionally shadows
                    cloud_mask = (scl == 8) | (scl == 9)  # SCL values 8 and 9 indicate clouds
                    if mask_shadows:
                        cloud_mask |= (scl == 3)  # SCL value 3 indicates shadows
                else:
                    print("SCL band not found!")

            if method in ('qa', 'standard'):
                print("Trying with quality band (QA60)")
                qa60 = None
                # Check for QA60 band
                if qa60_path: # Check if band provided in separate file
                    print(f"Using external QA60 band file: {qa60_path}")
                    with rasterio.open(qa60_path) as qa60_src:
                        qa60 = qa60_src.read(1)
                else: # Check if band idx is provided, otherwise tries to automatically find the band
                    if qa60_idx is None and 'QA60' in src.descriptions:
                        qa60_idx = src.descriptions.index('QA60') + 1 # Convert base-0 indexing to base-1 indexing (Rasterio works with base-1 indexing)
                    if qa60_idx:
                        qa60 = src.read(qa60_idx)
                        print("QA60 band found at index" + str(qa60_idx))

                if qa60 is not None:
                    if mask_shadows:
                        print("Warning: QA60 does not support shadow masking. Proceeding with cloud masking only.")
                    print("QA60 band found. Using it for cloud masking.")
                    print("Warning: QA60 is deprecated after 2022-01-25, results for images after that date could be wrong")
                    
                    # Define cloud and cirrus bit positions
                    CLOUD_BIT = 10
                    CIRRUS_BIT = 11

                    # Create cloud and cirrus masks
                    cloud_mask = ((qa60 & (1 << CLOUD_BIT)) == 0) & ((qa60 & (1 << CIRRUS_BIT)) == 0)
                else:
                    print("QA60 band not found!")

            if method in ('auto', 'omnicloudmask', 'shadows') and cloud_mask is None:
                print("Attempting omnicloudmask method.")
                
                # Check if index is provided, otherwise auto-detect Red band
                if red_idx is None and 'B4' in band_descriptions:
                    red_idx = band_descriptions.index('B4') + 1
                elif red_idx is None and 'B04' in band_descriptions:
                    red_idx = band_descriptions.index('B04')

                # Check if index is provided, otherwise auto-detect Green band
                if green_idx is None and 'B3' in band_descriptions:
                    green_idx = band_descriptions.index('B3') + 1
                elif green_idx is None and 'B03' in band_descriptions:
                    green_idx = band_descriptions.index('B03') + 1

                # Check if index is provided, otherwise auto-detect NIR band
                if nir_idx is None and 'B8A' in band_descriptions:
                    nir_idx = band_descriptions.index('B8A') + 1                    
                elif nir_idx is None and 'B8' in band_descriptions:
                    nir_idx = band_descriptions.index('B8') + 1
                elif nir_idx is None and 'B08' in band_descriptions:
                    nir_idx = band_descriptions.index('B08') + 1

                # If NIR is missing but Red & Green were found, look for Blue as a fallback
                if nir_idx is None and red_idx is not None and green_idx is not None:
                    print("NIR band not found. Attempting to use Blue band instead.")
                    if 'B2' in band_descriptions:
                        nir_idx = band_descriptions.index('B2') + 1
                    elif 'B02' in band_descriptions:
                        nir_idx = band_descriptions.index('B02') + 1

                # If Red, Green, and NIR/Blue are still missing, prompt the user
                if None in (red_idx, green_idx, nir_idx):
                    raise ValueError(
                        "Could not automatically determine the required bands (Red, Green, NIR/Blue). "
                        "Please call the function again and specify the band indexes manually."
                    )

                if red_idx is not None and green_idx is not None and nir_idx is not None:
                    # Read the specified bands
                    red_band = src.read(red_idx)
                    green_band = src.read(green_idx)
                    nir_band = src.read(nir_idx)

                    # Use the omnicloudmask function to generate a cloud mask
                    cloud_mask = omnicloudmask(red_band, green_band, nir_band, mask_shadows)

                else:
                    raise ValueError("Red, Green, and NIR band indices must be specified for the omnicloudmask method.")

            # Ensure a valid cloud mask is obtained
            if cloud_mask is None:
                raise ValueError("No valid method for cloud masking was found or specified.")

            # Mask the entire image using the cloud mask
            masked_data = []
            for band in range(1, src.count + 1):
                data = src.read(band)
                masked_band = np.where(cloud_mask, nodata_value, data)
                masked_data.append(masked_band)

            masked_data = np.array(masked_data)
            # Update metadata for saving
            metadata.update({"nodata": nodata_value, "dtype": 'float32'})

            # Write the masked image to output
            with rasterio.open(output_path, 'w', **metadata) as dest:
                dest.write(masked_data.astype('float32'))
                # Restore band descriptions
                for i, desc in enumerate(band_descriptions):
                    if desc:  # Only set descriptions if they exist
                        dest.set_band_description(i + 1, desc)

        print(f"Masked image saved to {output_path}.")
        return output_path  # Return the output file path
    
    except Exception as e:
        print(f"An error occurred: {e}")

def mask_clouds_landsat(image_path, output_path=None, method='auto', mask_shadows=False, nodata_value=np.nan, 
                        qa_pixel_path=None, qa_pixel_idx=None, confidence_threshold='High', 
                        red_idx=None, green_idx=None, nir_idx=None):
    """
    Masks clouds and optionally shadows in a Landsat raster image using various methods.

    Parameters:
        image_path (str): Path to the input multi-band raster image (float32 spectral bands).
        output_path (str, optional): Path to save the masked output raster. Defaults to same directory as input with '_masked' suffix.
        method (str): The method for masking ('auto', 'qa', 'omnicloudmask'). Defaults to 'auto'
        mask_shadows (bool): Whether to mask cloud shadows. Defaults to False
        qa_pixel_path (str, optional): Path to the separate QA_PIXEL raster file (if in a separate file).
        qa_pixel_idx (int, optional): Index of the QA_PIXEL band (1-based, if specified manually).
        red_idx (int, optional): Index of the red band (1-based, for omnicloudmask). Auto-detected if not provided.
        green_idx (int, optional): Index of the green band (1-based, for omnicloudmask). Auto-detected if not provided.
        nir_idx (int, optional): Index of the NIR band (1-based, for omnicloudmask). Auto-detected if not provided.

    Returns:
        str: The path to the saved masked output raster
    """
    try:
        # Set default output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_masked{ext}"

        # Open the raster file
        with rasterio.open(image_path) as src:
            metadata = src.meta.copy()
            band_descriptions = list(src.descriptions)  # Preserve band descriptions

            cloud_mask = None

            
            # Try QA_PIXEL method
            if method in ('auto', 'qa'):
                
                CLOUD_BIT = 3       # Cloud flag (bit 3)
                SHADOW_BIT = 4      # Cloud shadow flag (bit 4)
                CONFIDENCE_BITS = {
                    "cloud": (8, 9),  # Cloud confidence bits
                    "shadow": (10, 11)  # Cloud shadow confidence bits
                }
                CONFIDENCE_LEVELS = {
                    "none": 0,
                    "low": 1,
                    "medium": 2,
                    "high": 3
                }

                # Check for QA_PIXEL band
                try:
                    if qa_pixel_path: # Check if band provided in a different file
                        print(f"Using QA_PIXEL mask from {qa_pixel_path}")
                        with rasterio.open(qa_pixel_path) as qa_src:
                            qa_band = qa_src.read(1).astype(np.uint16)  # Read as uint16
                    else: # Check if band idx is provided, otherwise tries to automatically find the band
                        if qa_pixel_idx is None and 'QA_PIXEL' in src.descriptions:
                            qa_pixel_idx = src.descriptions.index('QA_PIXEL') + 1 # Convert base-0 indexing to base-1 indexing (Rasterio works with base-1 indexing)
                        if qa_pixel_idx:
                            qa_band = src.read(qa_pixel_idx)
                            print("QA_PIXEL band found at index" + str(qa_pixel_idx))

                    if qa_band is not None:
                        # Extract confidence threshold as integer
                        confidence_threshold_value = CONFIDENCE_LEVELS.get(confidence_threshold.lower(), 3)

                        # Cloud mask based on cloud flag
                        cloud_mask = (qa_band & (1 << CLOUD_BIT)) != 0  # Cloud bit

                        # This is setup for the future improvements of the QA_PIXEL, as stated in the last Landsat 8-9 Guide (May 2024)
                        # For now always leave default confidence_threshold='high' 
                        cloud_confidence = ((qa_band >> CONFIDENCE_BITS["cloud"][0]) & 3)  # Extract bits 8-9
                        cloud_mask |= cloud_confidence >= confidence_threshold_value

                        if mask_shadows:
                            # Shadow mask based on shadow flag
                            shadow_mask = (qa_band & (1 << SHADOW_BIT)) != 0  # Shadow bit

                            # This is setup for the future improvements of the QA_PIXEL, as stated in the last Landsat 8-9 Guide (May 2024)
                            # For now always leave default confidence_threshold='high'  
                            shadow_confidence = ((qa_band >> CONFIDENCE_BITS["shadow"][0]) & 3)  # Extract bits 10-11
                            shadow_mask |= shadow_confidence >= confidence_threshold_value

                            # Combine cloud and shadow masks
                            cloud_mask |= shadow_mask

                except Exception as e:
                    print(f"Warning: Failed to process QA_PIXEL band ({e})")

            #If QA method fails or is unavailable, use OmniCloudMask
            if cloud_mask is None and method in ('auto', 'omnicloudmask'):
                print("Attempting OmniCloudMask method...")

                # Check if index is provided, otherwise auto-detect Red band
                if red_idx is None and 'B4' in band_descriptions:
                    red_idx = band_descriptions.index('B4') + 1
                elif red_idx is None and 'SR_B4' in band_descriptions:
                    red_idx = band_descriptions.index('SR_B4') + 1

                # Check if index is provided, otherwise auto-detect Green band
                if green_idx is None and 'B3' in band_descriptions:
                    green_idx = band_descriptions.index('B3') + 1
                elif green_idx is None and 'SR_B3' in band_descriptions:
                    green_idx = band_descriptions.index('SR_B3') + 1

                # Check if index is provided, otherwise auto-detect NIR band
                if nir_idx is None and 'B5' in band_descriptions:
                    nir_idx = band_descriptions.index('B5') + 1
                elif nir_idx is None and 'SR_B5' in band_descriptions:
                    nir_idx = band_descriptions.index('SR_B5') + 1

                # If NIR is missing but Red & Green were found, look for Blue as a fallback
                if nir_idx is None and red_idx is not None and green_idx is not None:
                    print("NIR band not found. Attempting to use Blue band instead.")
                    if 'B2' in band_descriptions:
                        nir_idx = band_descriptions.index('B2') + 1
                    elif 'SR_B2' in band_descriptions:
                        nir_idx = band_descriptions.index('SR_B2') + 1

                # If Red, Green, and NIR/Blue are still missing, prompt the user
                if None in (red_idx, green_idx, nir_idx):
                    raise ValueError(
                        "Could not automatically determine the required bands (Red, Green, NIR/Blue). "
                        "Please call the function again and specify the band indexes manually."
                    )

                # Read the specified bands
                red_band = src.read(red_idx)
                green_band = src.read(green_idx)
                nir_band = src.read(nir_idx)  # This may be NIR or Blue

                # Compute cloud mask
                cloud_mask = omnicloudmask(red_band, green_band, nir_band, mask_shadows=mask_shadows)

            # Ensure a valid cloud mask is obtained
            if cloud_mask is None:
                raise ValueError("No valid method for cloud masking was found or specified.")

            # Mask the entire image using the cloud mask
            masked_data = []
            for band in range(1, src.count + 1):
                data = src.read(band)
                masked_band = np.where(cloud_mask, nodata_value, data)
                masked_data.append(masked_band)

            masked_data = np.array(masked_data)

            # Update metadata for saving
            metadata.update({"nodata": nodata_value, "dtype": 'float32'})

            # Write the masked image to output, preserving band descriptions
            with rasterio.open(output_path, 'w', **metadata) as dest:
                dest.write(masked_data.astype('float32'))

                # Restore band descriptions
                for i, desc in enumerate(band_descriptions):
                    if desc:  # Only set descriptions if they exist
                        dest.set_band_description(i + 1, desc)

        print(f"Masked image saved to {output_path}.")
        return output_path  # Return the output file path

    except Exception as e:
        print(f"An error occurred: {e}")

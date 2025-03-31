# Changelog 

## **[0.3.0] - 2025-03-31**
### ✨ Added
- Added water masking functionality using the NDWI index.
- New functions: `mask_water_S2` and `mask_water_landsat`.


## **[0.2.0] - 2025-02-27**
### 🛠️ Fixed
- Fixed bugs in `stack_bands` that caused it to write empty bands and crash in certain cases.
- Improved resolution handling in `stack_bands` to prevent incorrect scaling.
- Fixed an issue where the QA method was not applied correctly in auto mode for `mask_clouds_S2`.

## **[0.1.0] - 2025-02-02**
- Initial release of GeoPre

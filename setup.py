import re
from setuptools import setup, find_packages

# Read version dynamically from __init__.py
with open("src/geopre/__init__.py", "r") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

setup(
    name="geopre",
    version=version,  
    description="Preprocessing tools for satellite imagery analysis",
    long_description=open('README.md', encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MatteoGF/GeoPre',
    author='Matteo Gobbi Frattini, Liang Zhongyou',
    author_email='matteo.gf@live.it',
    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    keywords='geospatial preprocessing remote sensing satellite imagery',
    
    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    install_requires=[
        'omnicloudmask',
        'numpy',
        'rasterio',
        'geopandas',
        'pyproj',
        'rioxarray',
        'xarray',
        'shapely',
        'pathlib',
        'typing'
    ],

    python_requires=">=3.7, <3.12",  # Prevents Python 12+ compatibility issues

)

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:01:41 2025

@author: isaac
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Load Caribbean coordinate data (first column = lat, second column = lon)
coord_df = pd.read_csv("Caribbean\Caribbean_coord.csv", header=None, names=['lat', 'lon'])

# Convert coord_df into a GeoDataFrame (important for spatial operations)
gdf = gpd.GeoDataFrame(coord_df, geometry=gpd.points_from_xy(coord_df['lon'], coord_df['lat']), crs="EPSG:4326")


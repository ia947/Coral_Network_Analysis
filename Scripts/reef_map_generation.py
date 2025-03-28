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

# Set up a Cartopy map projection
proj = ccrs.PlateCarree()

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': proj})

# Add refined geographic features
ax.coastlines(resolution='50m', linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
ax.text(-92, 25, "Gulf of Mexico", transform=ccrs.PlateCarree(),
        fontsize=12, color='gray', alpha=0.7)
ax.text(-75, 15, "Caribbean Sea", transform=ccrs.PlateCarree(),
        fontsize=12, color='gray', alpha=0.7)
ax.text(-65, 23, "North Atlantic Ocean", transform=ccrs.PlateCarree(),
        fontsize=12, color='gray', alpha=0.7)


# Set an extent for the Caribbean region
ax.set_extent([-100, -50, 10, 30], crs=proj)

# Add gridlines with formatted labels
gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color="gray", alpha=0.7)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

# Plot the Caribbean points from the gdf
gdf.plot(ax=ax, marker='o', color='red', markersize=50, alpha=0.7, edgecolor='black', transform=proj, label="Caribbean Nodes")

# Add a legend
#ax.legend(loc="upper right", frameon=True, fontsize=12)

plt.show()

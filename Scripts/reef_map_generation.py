# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:01:41 2025

@author: isaac
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import cmocean
import numpy as np


##################################################
###### CREATE REGIONAL MAPS BY NODE LAT/LON ######
##################################################

def create_caribbean_node_map():
    # Load Caribbean coordinate data (first column = lat, second column = lon)
    caribbean_coord_df = pd.read_csv("Caribbean\Caribbean_coord.csv", header=None, names=['lat', 'lon'])

    # Convert coordinate df into GeoDataFrame
    gdf = gpd.GeoDataFrame(caribbean_coord_df, geometry=gpd.points_from_xy(caribbean_coord_df['lon'], caribbean_coord_df['lat']), crs="EPSG:4326")

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
    ax.set_extent([-100, -50, 7.5, 37.5], crs=proj)

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
    
    plt.show()
    
    return plt

create_caribbean_node_map()

def create_io_node_map():
    # Load Indian Ocean reef node data
    io_df = pd.read_csv("IO\IO_coord.csv")

    # Create a GeoDataFrame using the 'Longitude' and 'Latitude' columns
    io_gdf = gpd.GeoDataFrame(
        io_df, 
        geometry=gpd.points_from_xy(io_df['Longitude'], io_df['Latitude']),
        crs="EPSG:4326"
    )

    # Set up a Cartopy map projection
    proj = ccrs.PlateCarree()

    # Create the figure and axis for the Indian Ocean map
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': proj})

    # Add geographic features
    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')

    # Add labels for major seas/channels in the Indian Ocean region
    ax.text(77, -15, "Indian Ocean", transform=ccrs.PlateCarree(),
            fontsize=12, color='gray', alpha=0.7)

    ax.text(60, 17, "Arabian Sea", transform=ccrs.PlateCarree(),
            fontsize=12, color='gray', alpha=0.7)

    ax.text(70, 4.5, "Laccadive Sea", transform=ccrs.PlateCarree(),
            fontsize=12, color='gray', alpha=0.7)

    ax.text(83, 14, "Bay of Bengal", transform=ccrs.PlateCarree(),
            fontsize=12, color='gray', alpha=0.7)

    ax.text(36, -27, "Mozambique Channel", transform=ccrs.PlateCarree(),
            fontsize=12, color='gray', alpha=0.7)

    # Set an extent for the Indian Ocean region
    ax.set_extent([20, 110, -40, 30], crs=proj)

    # Add gridlines with formatted labels
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color="gray", alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    # Plot the Indian Ocean reef nodes
    io_gdf.plot(ax=ax, marker='o', color='red', markersize=50, alpha=0.7,
                edgecolor='black', transform=proj)

    plt.show()
    
    return plt

create_io_node_map()

def create_gbr_node_map(lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    
    # Read in the data
    gbr_df = pd.read_csv(r"GBR/gbr_gauges_utm56.csv")

    # Filter by coordinate bounds if provided
    if lat_min is not None:
        gbr_df = gbr_df[gbr_df['Lat'] >= lat_min]
    if lat_max is not None:
        gbr_df = gbr_df[gbr_df['Lat'] <= lat_max]
    if lon_min is not None:
        gbr_df = gbr_df[gbr_df['Lon'] >= lon_min]
    if lon_max is not None:
        gbr_df = gbr_df[gbr_df['Lon'] <= lon_max]

    # Convert filtered df into GeoDataFrame
    gbr_gdf = gpd.GeoDataFrame(
        gbr_df,
        geometry=gpd.points_from_xy(gbr_df['Lon'], gbr_df['Lat']),
        crs="EPSG:4326"
    )

    # Set up projection and figure
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': proj})

    # Add map features
    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')

    # Set extent for GBR region
    if not gbr_gdf.empty:
        minx, miny, maxx, maxy = gbr_gdf.total_bounds
        ax.set_extent([minx-1, maxx+1, miny-1, maxy+1], crs=proj)
    else:
        ax.set_extent([140, 155, -26, -8], crs=proj)

    # Gridlines
    gl = ax.gridlines(
        draw_labels=True, linestyle="--", linewidth=0.5,
        color="gray", alpha=0.7
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    # Plot the filtered reef nodes
    gbr_gdf.plot(
        ax=ax,
        marker='o', color='red', markersize=50,
        alpha=0.7, edgecolor='black',
        transform=proj
    )

    plt.show()
    return plt

create_gbr_node_map(lat_min=-22, lat_max=-14, lon_min=145, lon_max=152)

##################################################################
###### CREATE BATHYMETRIC MAP OF REGIONS (USING GEBCO DATA) ######
##################################################################

# Load GEBCO data
ds = xr.open_dataset('gebco_2024_tid\gebco_2024_sub_ice_topo\GEBCO_2024_sub_ice_topo.nc')
elevation = ds.elevation

# Define region bounds [lon_min, lon_max, lat_min, lat_max]
REGIONS = {
    "GBR": [142, 155, -25, -10],
    "IO": [35, 85, -30, 10],
    "Caribbean": [-100, -50, 7.5, 37.5]
}

# Color normalization parameters
vmin, vmax = -6000, 0
colombia_colours = ["#FFD700", "#003893", "#CE1126"]
cmap_colombia = LinearSegmentedColormap.from_list("colombia", colombia_colours, N=256)

def plot_gbr_features():
    """Create GBR tidal corridor/bathymetry map"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Data subset
    gbr_subset = elevation.sel(lon=slice(142, 155), lat=slice(-25, -10))
    
    # Main plot
    im = gbr_subset.plot.imshow(
        ax=ax,
        cmap=cmap_colombia,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )
    
    # Tidal corridors --> To be adjusted later
    #corridors = {
    #    'Hydrographers Passage': ([146, 147.5, 149], [-16.5, -17.5, -18.5]),
    #    'Pandora Entrance': ([148.5, 150, 151.5], [-18, -19, -20])
    #}
    
    #for name, coords in corridors.items():
    #    ax.plot(coords[0], coords[1], '--', color='#FFD700', lw=1.5,
    #            transform=ccrs.PlateCarree(), label=name)
    
    # Map elements
    ax.add_feature(cfeature.LAND, color='#8B4513', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    
    # Colourbar and labels
    plt.colorbar(im, label='Elevation (m)', ax=ax, shrink=0.6)
    
    return fig

plot_gbr_features()

def plot_caribbean_features():
    """Create Caribbean bathymetry map"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Data subset
    carib_subset = elevation.sel(lon=slice(-100, -50), lat=slice(7.5, 37.5))
    
    # Main plot
    im = carib_subset.plot.imshow(
        ax=ax,
        cmap=cmap_colombia,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )
    
    # Key features --> To be adjusted later
    #features = {
    #    'Pedro Bank': {'lon': [-78.5, -78.0, -77.5], 'lat': [16.8, 17.0, 16.8]},
    #    'Windward Passage': {'lon': [-74.5, -72.0], 'lat': [20.0, 20.0]}
    #}
    
    #colors = {'Pedro Bank': '#CE1126', 'Windward Passage': '#FFD700'}
    #for name, coords in features.items():
    #    ax.plot(coords['lon'], coords['lat'], '--', color=colors[name],
    #            lw=1.5, transform=ccrs.PlateCarree(), label=name)
    
    # Map elements
    ax.add_feature(cfeature.LAND, color='#8B4513', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    
    # Colourbar and labels
    plt.colorbar(im, label='Elevation (m)', ax=ax, shrink=0.6)
    
    return fig

plot_caribbean_features()

def plot_io_features():
    """Create Indian Ocean bathymetry map"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Data subset
    io_subset = elevation.sel(lon=slice(35, 85), lat=slice(-30, 10))
    
    # Main plot
    im = io_subset.plot.imshow(
        ax=ax,
        cmap=cmap_colombia,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )
    
    # Map elements
    ax.add_feature(cfeature.LAND, color='#8B4513', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    
    # Colourbar and labels
    plt.colorbar(im, label='Elevation (m)', ax=ax, shrink=0.6)
    
    return fig

plot_io_features()

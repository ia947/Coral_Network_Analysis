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
from matplotlib import colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


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

def create_gbr_node_map():
    # Read in  data
    gbr_df = pd.read_csv(r"GBR/gbr_gauges_utm56.csv")

    # Define subregions with their respective bounds
    regions = {
        'Grenville': {'lat': (-12, -10), 'lon': (142, 145)},
        'Cairns': {'lat': (-17, -16), 'lon': (145, 147)},
        'Swain': {'lat': (-22, -20), 'lon': (149, 153)}
    }

    # Function to assign subregions
    def get_subregion(row):
        for region, bounds in regions.items():
            lat_min, lat_max = bounds['lat']
            lon_min, lon_max = bounds['lon']
            if (lat_min <= row['Lat'] <= lat_max) and (lon_min <= row['Lon'] <= lon_max):
                return region
        return None

    gbr_df['Subregion'] = gbr_df.apply(get_subregion, axis=1)
    gbr_df = gbr_df.dropna(subset=['Subregion'])  # Keep only the three subregions

    # Convert to GeoDataFrame
    gbr_gdf = gpd.GeoDataFrame(
        gbr_df,
        geometry=gpd.points_from_xy(gbr_df['Lon'], gbr_df['Lat']),
        crs="EPSG:4326"
    )

    # Set up the plot
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': proj})
    ax.text(148.5, -15, "Coral Sea", transform=ccrs.PlateCarree(),
            fontsize=12, color='gray', alpha=0.7)

    # Add map features
    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')

    # Set dynamic extent based on data
    if not gbr_gdf.empty:
        minx, miny, maxx, maxy = gbr_gdf.total_bounds
        ax.set_extent([minx-1, maxx+1, miny-1, maxy+1], crs=proj)
    else:
        ax.set_extent([140, 155, -26, -8], crs=proj)

    # Configure gridlines
    gl = ax.gridlines(
        draw_labels=True, linestyle="--", linewidth=0.5,
        color="gray", alpha=0.7
    )
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Define colours for each subregion and plot
    colors = {'Grenville': 'red', 'Cairns': 'blue', 'Swain': 'green'}
    gbr_gdf['color'] = gbr_gdf['Subregion'].map(colors)

    gbr_gdf.plot(
        ax=ax,
        marker='o', color=gbr_gdf['color'], markersize=50,
        alpha=0.7, edgecolor='black', transform=proj
    )

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=region,
        markerfacecolor=color, markersize=10
    ) for region, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', title='Subregion')

    plt.show()
    return plt

# Generate the map
create_gbr_node_map()

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
    gbr_subset = elevation.sel(lon=slice(140, 155), lat=slice(-26, -8))
    
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
    
    # get the Gridliner and turn off the top & right labels
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
        
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

# Create a bathymetric graph of the major features of each region
# Windward Passage, Equatorial, and Hydrographer's Passage
def plot_bathymetric_profiles():
    """Create bathymetric profiles (Fig 1.4)"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    transect_params = {
        "GBR": {'lat': -18, 'label': 'Hydrographers Passage Transect'},
        "IO": {'lat': -5, 'label': 'Equatorial Transect'},
        "Caribbean": {'lat': 20, 'label': 'Windward Passage Transect'}
    }
    
    for (region, params), ax in zip(transect_params.items(), axes):
        # First select longitude range
        lon_slice = elevation.sel(lon=slice(*REGIONS[region][:2]))
        
        # Then find nearest latitude point
        transect = lon_slice.sel(lat=params['lat'], method='nearest')
        
        # Plot the transect
        transect.plot(ax=ax, color='#003893', lw=2)
        ax.invert_yaxis()
        ax.set_title(params['label'], fontsize=12)
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Longitude')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

plot_bathymetric_profiles()


####################################################
###### REGIONAL CURRENT MAPS USING HyCOM DATA ######
####################################################

CURRENT_REGIONS = {
    "GBR": [142, 155, -25, -10],
    "IO": [35, 85, -30, 10],
    "Caribbean": [260, 310, 7.5, 37.5]
}

def plot_region_currents(region_name, season=''):
    """Create current-only plots with coloured, larger arrows over a plain background"""
    # Setup figure and plain-white background
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_facecolor('white')
    region = CURRENT_REGIONS[region_name]

    # Load HYCOM dataset without decoding times to avoid calendar errors
    hycom_url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
    ds = xr.open_dataset(hycom_url, decode_times=False)

    # Spatial subset
    ds = ds.sel(lon=slice(region[0], region[1]), lat=slice(region[2], region[3]))

    # Extract u/v, collapse dimensions
    u_var = ds['water_u']; v_var = ds['water_v']
    if 'time' in u_var.dims:
        u_var = u_var.isel(time=0); v_var = v_var.isel(time=0)
    if 'depth' in u_var.dims:
        u_var = u_var.isel(depth=0); v_var = v_var.isel(depth=0)

    # Compute speed and subsample
    stride = 10
    u = u_var[::stride, ::stride]; v = v_var[::stride, ::stride]
    speed = np.sqrt(u**2 + v**2)

    # Add coastlines only
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # Plot larger arrows coloured by speed
    Q = ax.quiver(
        u.lon, u.lat, u, v, speed,
        cmap='cividis',      # colourblind-friendly
        scale=50,             # smaller scale -> larger arrows
        width=0.005,          # wider arrows
        headwidth=6,          # larger heads
        headlength=8,
        transform=ccrs.PlateCarree(),
        zorder=3
    )

    # Colourbar
    cbar = fig.colorbar(Q, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('Current Speed (m/s)', fontsize=12)

    # Quiver key
    #ax.quiverkey(Q, 0.88, 0.1, 1, '1 m/s', labelpos='E',
    #             coordinates='axes', fontproperties={'size': 10})

    # Title and gridlines
    #ax.set_title(f'{region_name} Surface Currents – {season}', fontsize=16, pad=12)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False; gl.right_labels = False

    return fig

for region, season in zip(["GBR", "IO", "Caribbean"], ["Summer", "Monsoon", "Dry"]):
    fig = plot_region_currents(region, season)
    #plt.savefig(f'{region}_currents_only.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


####################################################
###### CREATE CUSTOM GLOBAL REEF COVERAGE MAP ######
####################################################

def plot_global_coral_distribution(shp_path, buffer_deg=0.05):
    # Load  Natural Earth reefs shapefile and reproject to WGS84
    reefs = gpd.read_file(shp_path).to_crs(epsg=4326)

    # Buffer each polygon to make them visually thicker on a global map
    reefs['geometry'] = reefs.geometry.buffer(buffer_deg)

    # Create a global PlateCarree map
    fig, ax = plt.subplots(
        figsize=(14, 7),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    ax.set_global()  # show entire world
    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')

    # Plot all reef polygons with thicker edges
    reefs.plot(
        ax=ax,
        facecolor='coral',
        edgecolor='darkred',
        linewidth=1.0,      # thicker border
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=2
    )
    
    plt.tight_layout()

    return fig, ax

shapefile_path = 'ne_10m_reefs/ne_10m_reefs.shp'
fig, ax = plot_global_coral_distribution(shapefile_path, buffer_deg=0.75)
plt.show()
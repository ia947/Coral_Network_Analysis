# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:01:43 2025

@author: isaac
"""

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


############################################
###### LOAD CSV FILES FOR EACH REGION ######
############################################

# Set file paths
base_path = r"C:\Users\isaac\SynologyDrive\Documents\University of York\BSc (Hons) Environmental Geography\3rd Year (2024-2025)\Dissertation\Code and Data"

# Map all GBR csv files
GBR_files = [f for f in os.listdir(base_path) if f.startswith("GBR_") and f.endswith("_centrality_measures.csv")]

# Create dictionary with the proper keys
IO_and_C_centrality_measures_files = {
    "IO": os.path.join(base_path, "IO_centrality_measures.csv"),
    "Caribbean": os.path.join(base_path, "Caribbean_centrality_measures.csv"),
    }

for file in GBR_files:
    key = file.replace("_centrality_measures.csv", "") # To remove the file suffix
    IO_and_C_centrality_measures_files[key] = os.path.join(base_path, file)
    
# Load all data into dictionary
dataframes = {region: pd.read_csv(file) for region, file in IO_and_C_centrality_measures_files.items()}

# Then add a region identifier to each dataframe
for region, df in dataframes.items():
    # Assign a region name (GBR, IO, or Caribbean)
    if "GBR" in region:
        df["Region"] = "GBR"
        df["Condition"] = region.replace("GBR_", "")  # To store the specific area (e.g., "tides-only_Cairns")
    elif "IO" in region:
        df["Region"] = "IO"
        df["Condition"] = "IO"
    elif "Caribbean" in region:
        df["Region"] = "Caribbean"
        df["Condition"] = "Caribbean"

df_all = pd.concat(dataframes.values(), ignore_index=True)

# Display basic information
print(df_all.head())


###############################
###### NORMALITY TESTING ######
###############################

metrics = ["Degree Centrality", "Network Centralisation", "Closeness Centrality", "Betweenness Centrality",
           "Eigenvector Centrality", "Harmonic Centrality", "Clustering Coefficient", "Graph Density",
           "Rich Club Coefficient", "Transitivity", "Local Efficiency"]

for metric in metrics:
    print(f"\nShapiro-Wilk test for normality - {metric}:")
    for region in dataframes.keys():
        stat, p = stats.shapiro(df_all[df_all["Region"] == region][metric])
        print(f"{region}: W={stat:.3f}, p={p:.3f}")
        if p < 0.05:
            print(f"    {metric} is **not normally distributed** in {region}")
        else:
            print(f"    {metric} is normally distributed in {region}")

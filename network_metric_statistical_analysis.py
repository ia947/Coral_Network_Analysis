# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:01:43 2025

@author: isaac
"""

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import ast


############################################
###### LOAD CSV FILES FOR EACH REGION ######
############################################

# Set base directory
base_path = r"C:\Users\isaac\SynologyDrive\Documents\University of York\BSc (Hons) Environmental Geography\3rd Year (2024-2025)\Dissertation\Code and Data"

# Initialise file paths dictionary
network_metrics_files = {}

# Detect and add GBR network metric files
for file in os.listdir(base_path):
    if file.endswith("_network_metrics.csv"):
        file_path = os.path.join(base_path, file)

        if file.startswith("GBR_"):  
            key = file.replace("_network_metrics.csv", "")  # Remove suffix
            network_metrics_files[key] = file_path  # Store with original name

        elif file.startswith("IO_network_metrics"):
            network_metrics_files["IO"] = file_path

        elif file.startswith("Caribbean_network_metrics"):
            network_metrics_files["Caribbean"] = file_path

# Check if all expected regions are found
expected_regions = ["IO", "Caribbean"] + [key for key in network_metrics_files if key.startswith("GBR")]
missing_regions = [region for region in expected_regions if region not in network_metrics_files]

if missing_regions:
    raise FileNotFoundError(f"Missing network metrics files for: {missing_regions}")

# Load data into dictionary
dataframes = {region: pd.read_csv(file) for region, file in network_metrics_files.items()}

# Ensure DataFrames are not empty
for region, df in dataframes.items():
    if df.empty:
        raise ValueError(f"Dataset for {region} is empty. Check {network_metrics_files[region]}.")

# Add region identifiers
for region, df in dataframes.items():
    if "GBR" in region:
        df["Region"] = region
        df["Condition"] = region.replace("GBR_", "")  # Extract condition
    else:
        df["Region"] = region  # IO or Caribbean
        df["Condition"] = region  # Maintain consistency

# Combine into a single DataFrame
df_all = pd.concat(dataframes.values(), ignore_index=True)

# Ensure dataset isn't empty after merging
if df_all.empty:
    raise ValueError("Merged dataset is empty. Verify CSV files.")

##################################
###### NORMALITY TESTING #########
##################################

metrics = [
    "Degree Centrality", "Network Centralisation", "Closeness Centrality", "Betweenness Centrality",
    "Eigenvector Centrality", "Harmonic Centrality", "Clustering Coefficient", "Graph Density",
    "Rich Club Coefficient", "Transitivity", "Local Efficiency"
    ]

# Check if all expected metrics exist
missing_metrics = [col for col in metrics if col not in df_all.columns]
if missing_metrics:
    raise ValueError(f"Missing expected columns in dataset: {missing_metrics}")

# Drop rows with missing values
df_all.dropna(subset=metrics, inplace=True)

normality_results = {}

for metric in metrics:
    print(f"\nNormality test for {metric}:\n" + "-" * 40)

    normality_results[metric] = {}

    for region in dataframes.keys():
        data = df_all[df_all["Region"] == region][metric].dropna()
        
        if len(data) == 0:
            print(f"Skipping {region} for {metric} due to missing data.")
            continue

        # Convert any non-numeric values to NaN
        data = pd.to_numeric(data, errors='coerce')

        # Handle dictionary-like strings if present
        if isinstance(data.iloc[0], str) and data.iloc[0].startswith('{'):
            data = data.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Remove rows with NaN values after conversion
        data = data.dropna()

        # Skip region-metric combinations with no valid data
        if data.empty:
            print(f"Skipping {region} for {metric} (no valid data after conversion).")
            continue

        # Standardisation
        std_dev = data.std()
        if std_dev == 0 or np.isnan(std_dev):
            print(f"Skipping {region} for {metric} (constant values).")
            continue

        standardised_data = (data - data.mean()) / std_dev

        # Normality test
        if len(data) > 500:
            stat, p = stats.kstest(standardised_data, 'norm')
            test_used = "Kolmogorov-Smirnov"
        else:
            stat, p = stats.shapiro(data)
            test_used = "Shapiro-Wilk"

        normality_results[metric][region] = p

        print(f"{region}: {test_used} test - W={stat:.3f}, p={p:.3f}")
        if p < 0.05:
            print(f"    **{metric} is NOT normally distributed** in {region}")
        else:
            print(f"    {metric} is normally distributed in {region}")


###############################################
###### ANOVA / KRUSKAL-WALLIS TESTING #########
###############################################













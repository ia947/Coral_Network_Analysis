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
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import plotly.figure_factory as ff
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

###############################
###### NORMALITY TESTING ######
###############################

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

normality_results = []

for metric in metrics:
    print(f"\nNormality test for {metric}:\n" + "-" * 40)
    
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
            
        # Store results in normality_results list
        normality_results.append({
            "Metric": metric,
            "Region": region,
            "Test": test_used,
            "Statistic": stat,
            "p-value": p,
            "Normally Distributed": "Yes" if p >= 0.05 else "No"
            })
        
        print(f"{region}: {test_used} test - W={stat:.3f}, p={p:.3f}")
        if p < 0.05:
            print(f"    **{metric} is NOT normally distributed** in {region}")
        else:
            print(f"    {metric} is normally distributed in {region}")

# Convert results list into df and save to csv
normality_df = pd.DataFrame(normality_results)
normality_df.to_csv(r"normality_test_results.csv")

############################################
###### ANOVA / KRUSKAL-WALLIS TESTING ######
############################################

# Define testable metrics for comparison
comparison_metrics = ["Network Centralisation", "Graph Density", "Degree Centrality"]

ANOVA_KruskallWallis_results = []

# Perform one-way ANOVA or Kruskal-Wallis depending on normality
for metric in comparison_metrics:
    print(f"\nComparing {metric} between regions:")
    
    # Group data by region
    data_by_region = [df_all[df_all["Region"] == region][metric].dropna() for region in dataframes.keys()]
    
    # Check normality of the metric in each region
    normal_regions = [
        region for region in dataframes.keys() 
        if any(result['Metric'] == metric and result['Region'] == region and result['p-value'] >= 0.05 for result in normality_results)
    ]
    non_normal_regions = [
        region for region in dataframes.keys() 
        if any(result['Metric'] == metric and result['Region'] == region and result['p-value'] < 0.05 for result in normality_results)
    ]
    
    if all(region in normal_regions for region in dataframes.keys()):
        # All regions are normal, apply ANOVA
        stat, p = stats.f_oneway(*data_by_region)
        test_used = "ANOVA"
    else:
        # At least one region is non-normal, apply Kruskal-Wallis
        stat, p = stats.kruskal(*data_by_region)
        test_used = "Kruskal-Wallis"
    
    print(f"{test_used} test: statistic={stat:.3f}, p={p:.3f}")
    if p < 0.05:
        print(f"    Significant difference found in {metric}")
    else:
        print(f"    No significant difference found in {metric}")
        
    # Store ANOVA/Kruskall-Wallis results
    ANOVA_KruskallWallis_results.append({
        "Metric": metric,
        "Test": test_used,
        "Statistic": stat,
        "p-value": p,
        "Significant difference": "Yes" if p < 0.05 else "No"
        })

    # If significant, perform post-hoc test (e.g., Tukey's HSD)
    if p < 0.05:
        # For ANOVA
        if test_used == "ANOVA":
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            all_data = pd.concat([df_all[["Region", metric]] for df_all in dataframes.values()])
            tukey_result = pairwise_tukeyhsd(all_data[metric], all_data["Region"])
            print("\nPost-hoc Tukey's HSD results:")
            print(tukey_result.summary())
            
            # Store Tukey's HSD results
            tukey_results = tukey_result.summary().data
            for row in tukey_results[1:]:
                ANOVA_KruskallWallis_results.append({
                    "Metric": metric,
                    "Test": "Tukey's HSD",
                    "Pairwise Comparison": f"{row[0]} vs {row[1]}",
                    "Statistic": row[2],
                    "p-value": row[3],
                    "Significant difference": "Yes" if row[3] < 0.05 else "No"
                    })

# Convert results list into df and save to csv
ANOVA_KruskallWallis_df = pd.DataFrame(ANOVA_KruskallWallis_results)
ANOVA_KruskallWallis_df.to_csv(r"ANOVA_KruskallWallis_results.csv", index=False)


##########################################
###### PRINCIPAL COMPONENT ANALYSIS ######
##########################################

# Standardise results for PCA
metrics_data = df_all[metrics].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(metrics_data)

# Perform PCA
pca = PCA()
pca.fit(scaled_data)

# Visualise explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(metrics) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('PCA Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.show()

# Biplot
components = pca.components_
plt.figure(figsize=(10, 6))
sns.heatmap(components, cmap='coolwarm', xticklabels=metrics, yticklabels=[f"PC{i+1}" for i in range(len(components))])

plt.title('PCA Loadings')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels to 45 degrees for readability
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Extract principal components
pca_data = pca.transform(scaled_data)
df_pca = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(len(pca_data[0]))])

# Add region info to PCA data
df_pca["Region"] = df_all["Region"]

# Get the loadings for PC1 (first row of pca.components_)
pc1_loadings = pca.components_[0]

# Create a DataFrame to show the loadings of each metric for PC1 --> adjust for other PCs if necessary
pc1_loadings_df = pd.DataFrame(pc1_loadings, index=metrics, columns=['PC1'])
print(pc1_loadings_df)


#######################################################################
###### HIERARCHICAL CLUSTERING (W. SILHOUETTE AND ELBOW METHODS) ######
#######################################################################

# Elbow Method
wcss = []  # List to store within-cluster sum of squares (WCSS) for each k
for k in range(2, 11):  # Check for k from 2 to 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Silhouette Method
sil_scores = []  # List to store silhouette scores for each k
for k in range(2, 11):  # Check for k from 2 to 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    sil_score = silhouette_score(scaled_data, cluster_labels)
    sil_scores.append(sil_score)

# Plot the silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), sil_scores, marker='o', linestyle='--')
plt.title('Silhouette Scores for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Hierarchical Clustering in Principal Component space due to high dimensionality
# Use PC1 and PC2 --> 2 dimensions
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Perform hierarchical clustering
linkage_matrix = linkage(reduced_data, method='ward')

# Assign clusters
num_clusters = 3  # Adjust as needed
cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

# Define distinct colors for each cluster
palette = sns.color_palette("tab10", num_clusters)  # Uses discrete colors
colors = [palette[label - 1] for label in cluster_labels]

# Plot the clusters in PCA space
plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, edgecolors='k')

# Create a legend for clusters
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=f'Cluster {i+1}') 
           for i in range(num_clusters)]
plt.legend(handles=handles, title="Clusters")

# Labels and title
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Hierarchical Clustering in PCA Space")
plt.show()

# Add cluster labels to the dataframe
df_all['Cluster'] = cluster_labels


##########################################
###### PEARSON/SPEARMAN CORRELATION ######
##########################################

# Calculate correlation matrix for metrics
correlation_matrix = df_all[metrics].corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=0.5, vmin=0, vmax=1)
plt.title('Correlation Matrix of Network Metrics')
plt.xticks(rotation=45, ha='right')
plt.show()

# For non-parametric correlation (Spearman)
correlation_matrix_spearman = df_all[metrics].corr(method='spearman')
sns.heatmap(correlation_matrix_spearman, annot=True, cmap='viridis', linewidths=0.5, vmin=0, vmax=1)
plt.title('Spearman Correlation Matrix of Network Metrics')
plt.xticks(rotation=45, ha='right')
plt.show()


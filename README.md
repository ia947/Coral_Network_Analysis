# Coral Network Analysis
This repository contains scripts for network analysis of coral reef connectivity, developed for BSc Environmental Geography dissertation at the University of York.

## **Overview**
Coral reef ecosystems are crucial biodiversity hotspots, and their connectivity plays a key role in their resilience and conservation. This study compares network structures of coral reefs in three regions:
- **Great Barrier Reef (GBR)** - Tidal corridors and wind conditions
- **Indian Ocean (IO)** - Basin-scale connectivity
- **Caribbean** - Island hub-and-spoke network structure

By analysing connectivity matrices derived from **biophysical larval dispersal models**, I identify key patterns in reef connectivity and their implications for conservation.

## **Aims**
The study aims to:
1. Ascertain whether there is a difference in network characterisation between coral reefs in the GBR, IO, and Caribbean (i.e. are they hub-and-spoke, or are they mesh).
2. Calculate whether the density of reefs and reef connectivity differs between the GBR, IO, and Caribbean.
3. Identify and analyse metrics that provide insights into the styles of connectivity between different regions.

## **Methods**
A **graph-theoretic approach** is employed to analyse connectivity, where:
- **Nodes** represent coral reefs
- **Edges** represent larval dispersal strength (weighted connections)

**Data sources:** Connectivity matrices were generated from biophysical Lagrangian Particle Tracking (LPT) models.

### **Key Metrics Computed in `coral_network_analysis.py`**
Using 'NetworkX', the following metrics are calculated:
- **Degree Centrality**: Importance of a reef based on direct connections.
- **Network Centralisation**: Overall reliance of the network on individual nodes.
- **Closeness Centrality**: The higher the value, the closer it is to all other nodes.
- **Betweenness Centrality**: Measures the influence a node has on controlling connections to other nodes.
- **Eigenvector Centrality**: Importance of a node based on its connection to other important nodes.
- **Harmonic Centrality**: Sum of the inverse of all shortest paths to other nodes.
- **Clustering Coefficient**: Shows how connected a node's immediate neighbours are.
- **Graph Density**: Ratio of the number of edges to the maximum possible number of edges.
- **Rich Club Coefficient**: Measures the tendency of well-connected nodes to connect with other well-connected nodes.
- **Transitivity**: How likely that adjacent nodes in a network are connected.
- **Local Efficiency**: Assesses the interconnectivity of a node's immediate neighbours, indicating local robustness.

### **Statistical Tests Performed in `network_metric_statistical_analysis.py`**
- **Normality Testing**: Shapiro-Wilk (for sample sizes ≤ 500) and Kolmogorov-Smirnov (for larger samples).
- **Comparative Analysis**: ANOVA or Kruskal-Wallis testing to determine the differences in regional distributions.
- **Tukey's HSD**: Following ANOVA/Kruskal-Wallis if significant differences are found, for pairwise comparisons.
- **Principal Component Analysis (PCA)**: To reduce dimensionality and highlight key components explaining data variance.
- **Hierarchical Clustering**: Including the **Elbow Method** and **Silhouette Score** on PCA-reduced data to group metrics by their explained variance.
- **Pearson/Spearman Correlation**: To assess linear/monotonic relationships between all metrics.

### **Python Libraries Used**
- **NetworkX** for graph analysis
- **NumPy** and **Pandas** for data handling
- **Matplotlib** and **Seaborn** for data visualisation

## **Repository Structure**
This repository is organised as follows:
```plaintext
Code and Data/  
│── Caribbean/                 # Data and analysis specific to the Caribbean reefs  
│── Coral_Network_Analysis/    # Main directory for network analysis scripts  
│── Data/                      # Raw and processed datasets  
│── GBR/                       # Data and analysis specific to the Great Barrier Reef  
│── IO/                        # Data and analysis specific to the Indian Ocean  
│── Metric Distributions and Statistical Analysis/  # Statistical outputs and metric distributions  
│── Network Graph Outputs/     # Visualisation outputs for network analysis  
│── Scripts/                   # Python scripts for analysis  
│── README.md                  # Project documentation  
```  

### **Folder Descriptions**  
- **`Caribbean/`** – Contains data and analysis results for the Caribbean region.  
- **`Coral_Network_Analysis/`** – Main directory for network analysis scripts and related files.  
- **`Data/`** – Stores raw and processed datasets used for connectivity analysis.  
- **`GBR/`** – Contains data and analysis specific to the Great Barrier Reef region.  
- **`IO/`** – Contains data and analysis specific to the Indian Ocean region.  
- **`Metric Distributions and Statistical Analysis/`** – Outputs from statistical analysis of network metrics.  
- **`Network Graph Outputs/`** – Contains visual representations of coral network structures.  
- **`Scripts/`** – Python scripts for running connectivity analysis and statistical computations.  
- **`README.md`** – Project documentation.  


## Licensing

- **Code**: [MIT License](LICENSE_CODE.md)
- **Data**: [CC-BY-SA 4.0 License](LICENSE_DATA.md)

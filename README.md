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

### **Other Scripts**:
- `reef_map_generation.py` is a geospatial mapping toolkit for the contexual bathymetric and ocean currents governing each region. These maps are created using mutliple data layers, including HyCOM currents (https://www.hycom.org/dataserver), GEBCO bathymetry data (https://www.gebco.net/data-products/gridded-bathymetry-data), and Natural Earth global coral reef coverage (ne_10m_reefs).
- `example_network_graph_plotting.py` is a conceptual network diagram generator. The outputs of this script are very simple edge-node networks for those less familiar with the principles governing network analysis. A basic 2-node network, a self-edge network, and hub-and-spoke versus mesh network are visualised.
- `pld_phase_plot.py` creates a temporal phase diagram for coral larval development, illustrating the different phases experienced throughout the pelagic larval duration. The parameters for each phase can be changed to reflect the specific phases of different larval species. In this current script, only a general plot is produced, with uncertainty bars to show potential ranges for other species.
- `IOnetcdf_to_csv.py` quickly handles the Indian Ocean's netCDF file to convert it into a csv, which is more computationally efficient in the full analysis.

### **Python Libraries Used**
#### Core Analysis
| Library | Purpose | Version |
|---------|---------|---------|
| `NetworkX` | Network analysis and metrics | ≥3.0 |
| `NumPy` | Matrix operations and math | ≥1.23 |
| `Pandas` | Data manipulation and I/O | ≥2.0 |

#### Visualization
| Library | Purpose | Version |
|---------|---------|---------|
| `Matplotlib` | Base plotting system | ≥3.7 |
| `Cartopy` | Geospatial mapping | ≥0.21 |
| `GeoPandas` | Spatial data handling | ≥0.13 |
| `cmocean` | Oceanographic colormaps | ≥2.0 |
| `graphviz` | Graph layout algorithms | ≥0.20 |

#### Geospatial Processing
| Library | Purpose | Version |
|---------|---------|---------|
| `xarray` | NetCDF/GRIB data handling | ≥2023.12 |
| `PyProj` | Coordinate transformations | Built-in Cartopy |

#### Advanced Statistics
| Library | Purpose | Version |
|---------|---------|---------|
| `SciPy` | Statistical tests and math | ≥1.11 |
| `scikit-learn` | PCA and clustering | ≥1.3 |

#### Utilities
| Library | Purpose | Version |
|---------|---------|---------|
| `tqdm` | Progress bars | (Optional) |
| `pyogrio` | Fast shapefile I/O | (GeoPandas dependency) |

**Install all requirements:**
```bash
conda install -c conda-forge networkx pandas matplotlib cartopy geopandas xarray cmocean scipy scikit-learn graphviz
pip install pyogrio tqdm
```

## **Repository Structure**
This repository is organised as follows:
```plaintext
Code and Data/  
│── Caribbean/
│── GBR/
│── IO/
│── Metric Distributions and Statistical Analysis/
│── Network Graph Outputs/
│── Scripts/
│── LICENCE_CODE.md
│── LICENCE_DATA.md
│── README.md
```  

### **Folder Descriptions**  
- **`Caribbean/`** – Contains data and analysis results for the Caribbean region. 
- **`GBR/`** – Contains data and analysis specific to the Great Barrier Reef region.  
- **`IO/`** – Contains data and analysis specific to the Indian Ocean region.  
- **`Metric Distributions and Statistical Analysis/`** – Outputs from statistical analysis of network metrics.  
- **`Network Graph Outputs/`** – Contains visual representations of coral network structures.  
- **`Scripts/`** – Python scripts for running connectivity analysis and statistical computations.
- **`LICENCE_CODE.md/`** - MIT licence for code.
- **`LICENCE_DATA.md/`** - CC-BY-SA 4.0 licence for data.
- **`README.md`** – Project documentation.  


## Licensing

- **Code**: [MIT License](LICENSE_CODE.md)
- **Data**: [CC-BY-SA 4.0 License](LICENSE_DATA.md)

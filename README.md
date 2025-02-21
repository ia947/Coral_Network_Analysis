# Coral Network Analysis
this repository contains scripts for network analysis of coral reef connectivity, as part of my BSc Environmental Geography dissertation at the University of York.

## **Overview**
Coral reef ecosystems are crucial biodiversity hotspots, and their connectivity plays a key role in their resilience and conservation. This study compares network structures of coral reefs in three regions:
- **Great Barrier Reef (GBR)**
- **Indian Ocean (IO)**
- **Caribbean**

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

### **Key Metrics Computed in 'coral_network_analysis.py'**
Using 'NetworkX', the following metrics are calculated:
- **Degree Centrality**: Importance of a reef based on direct connections.
- **Network Centralisation**: Overall reliance of the network on individual nodes.
- **Closeness Centrality**: The higher the value, the closer it is to all other nodes.
- **Betweenness Centrality**:
- **Eigenvector Centrality**:
* Harmonic Centrality
* Clustering Coefficient
* Graph Density
* Rich Club Coefficient
* Transitivity
* Local Efficiency


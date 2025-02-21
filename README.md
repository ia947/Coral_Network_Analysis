# Coral Network Analysis
Network analysis scripts for coral source-sink matrices, as part of BSc Environmental Geography dissertation.

## Aims and Objectives of Study
My dissertation aims to compare how the health and dynamics of coral ecosystems differ between the Great Barrier Reef (GBR), Indian Ocean, (IO), and Caribbean.

The general aims are:
1. Ascertain whether there is a difference in network characterisation between coral reefs in the GBR, IO, and Caribbean (i.e. are they hub-and-spoke, or are they mesh).
2. Calculate whether the density of reefs and reef connectivity differs between the GBR, IO, and Caribbean.
3. Identify and analyse metrics that provide insights into the styles of connectivity between different regions.

## Computational Methods Used
Connectivity matrices (see each respective region's folder in this repository for each csv file) were used to represent reefs (nodes) and larval dispersal strengths (weighted edges). Each of these matrices were derived using biophysical larval dispersal modelling, using Lagrangian Particle Tracking.
NetworkX was the primary Python module used in coral_network_analysis.py to calculate each network metric:
* Degree Centrality
* Network Centralisation
* Closeness Centrality
* Betweenness Centrality
* Eigenvector Centrality
* Harmonic Centrality
* Clustering Coefficient
* Graph Density
* Rich Club Coefficient
* Transitivity
* Local Efficiency


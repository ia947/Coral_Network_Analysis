# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:39:13 2024

@author: isaac
"""

#Import initial modules
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

#################################
###### DATA PRE-PROCESSING ######
#################################

# Function to read connectivity matrix
#filename = r"GBR\wind_and_tides\grenville\connectivity_decimal.csv"
filename = r"IO\IO_single_step_explicit_mean_connectivity_matrix.csv"
#filename = r"Caribbean\D_Caribbean_revised.npy"

def read_adjacency_matrix(filename):
    #adjacency_matrix = Dataset(filename, mode='r')
    adjacency_matrix = np.genfromtxt(filename, delimiter=',', skip_header=0)
    #adjacency_matrix = np.load(filename)
    return adjacency_matrix

# Function to create directed graph from connectivity matrix
def create_adjacency_matrix_graph(adjacency_matrix):
    G = nx.DiGraph()
    num_nodes = len(adjacency_matrix)
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):  # Include all connections (i -> j)
            if adjacency_matrix[i][j] > 0:
                G.add_edge(i, j, weight=adjacency_matrix[i][j])
    return G

####################################
###### NETWORK VISUALISATION #######
####################################

# Function to draw the directed graph
def draw_graph(G, use_graphviz=False):
    if use_graphviz:
        try:
            pos = graphviz_layout(G, prog='sfdp')  # Use sfdp for sprawling layout
        except ImportError:
            print("Graphviz is not installed. Falling back to spring_layout.")
            pos = nx.spring_layout(G, seed=42, k=0.3, iterations=100)
    else:
        pos = nx.spring_layout(G, seed=42, k=0.3, iterations=100)  # Adjust k for spread
    
    # Compute degree centrality for coloring
    degree_centrality = nx.degree_centrality(G)
    centrality_values = list(degree_centrality.values())
    max_centrality = max(centrality_values)
    min_centrality = min(centrality_values)
    
    # Node properties
    fixed_node_size = 1000  # Set a fixed size for all nodes
    node_color = [degree_centrality[node] for node in G.nodes()]
    cmap = plt.cm.viridis  # Use a perceptually uniform colormap
    
    # Edge properties
    edge_weights = nx.get_edge_attributes(G, 'weight')
    edge_width = [0.5 + 3 * edge_weights[edge] for edge in G.edges()]  # Scale edge width by weight
    
    # Draw graph
    plt.figure(figsize=(15, 15))
    nx.draw(
        G,
        pos,
        node_size=fixed_node_size,
        node_color=node_color,
        cmap=cmap,
        edge_color='gray',
        width=edge_width,
        alpha=0.7,
        arrows=True,
        arrowsize=10
    )
    
    # Add colorbar for node centrality
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_centrality, vmax=max_centrality))
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.03, pad=0.04)
    cbar.set_label("Degree Centrality", fontsize=14)
    
    plt.show()

adjacency_matrix = read_adjacency_matrix(filename)
G = create_adjacency_matrix_graph(adjacency_matrix)
#draw_graph(G, use_graphviz=True)


##################################
###### CENTRALITY MEASURES #######
##################################

# Function to compute centrality measures
def compute_centralities(G, output_filename):
    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    # Closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    # Betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    # Eigenvector centrality
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    # Harmonic centrality
    harmonic_centrality = nx.harmonic_centrality(G)
    # Clustering coefficient
    clustering_coefficient = nx.clustering(G.to_undirected())

    centrality_df = pd.DataFrame({
        'Node': list(G.nodes),
        'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
        'Closeness Centrality': [closeness_centrality[node] for node in G.nodes()],
        'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()],
        'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()],
        'Harmonic Centrality': [harmonic_centrality[node] for node in G.nodes()],
        'Clustering Coefficient': [clustering_coefficient[node] for node in G.nodes()]
    })
    
    # Write centrality dataframe as CSV
    centrality_df.to_csv(output_filename, index=False)
    return centrality_df

summary_stats = compute_centralities(G, 'IO_centrality_measures.csv')
print(summary_stats)

# Next to run the script for each location to produce centrality csv files for each

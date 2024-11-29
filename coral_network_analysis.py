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

#################################
###### DATA PRE-PROCESSING ######
#################################

# Function to read connectivity matrix
#filename = r"GBR\wind_and_tides\grenville\connectivity_decimal.csv"
filename = r"Caribbean\D_Caribbean_revised.npy"

def read_adjacency_matrix(filename):
    #adjacency_matrix = np.load(filename)
    adjacency_matrix = np.genfromtxt(filename, delimiter=',', skip_header=0)
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
draw_graph(G, use_graphviz=True)


###############################
###### DEGREE CENTRALITY ######
###############################



##################################
###### CLOSENESS CENTRALITY ######
##################################



####################################
###### BETWEENNESS CENTRALITY ######
####################################



####################################
###### EIGENVECTOR CENTRALITY ######
####################################



#################################
###### HARMONIC CENTRALITY ######
#################################



####################################
###### CLUSTERING COEFFICIENT ######
####################################



######################################
###### GRAPH DENSITY (DIRECTED) ######
######################################



###################################
###### RICH CLUB COEFFICIENT ######
###################################



##########################
###### TRANSITIVITY ######
##########################



##############################
###### LOCAL EFFICIENCY ######
##############################



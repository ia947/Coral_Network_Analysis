# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:39:13 2024

@author: isaac
"""

#Import initial modules
import argparse
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
#import geopandas as gpd
import matplotlib.pyplot as plt
import os
import tarfile

#################################
###### DATA PRE-PROCESSING ######
#################################

## *Make sure all adjacency matrices are in .npy or .csv before reading* ##

filename = r"Caribbean\D_Caribbean_revised.npy"

def read_adjacency_matrix(filename):
    #adjacency_matrix = np.genfromtxt(filename, delimiter=',', skip_header=1)
    adjacency_matrix = np.load(filename)
    return adjacency_matrix

# Creating simple graph for the adjacency matrix
def create_adjacency_matrix_graph(adjacency_matrix):
    G = nx.DiGraph()
    num_nodes = len(adjacency_matrix)
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j] > 0:
                G.add_edge(i,j)
    return G

#################################
###### DATA VISUALIZATION #######
#################################

def draw_graph(G, pos=None):
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=0.15)

    # Compute degree centrality
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    centrality_values = list(eigenvector_centrality.values())
    max_centrality = max(centrality_values)
    min_centrality = min(centrality_values)
    
    # Map the centrality values to a colour scale
    node_color = [eigenvector_centrality[node] for node in G.nodes()]
    cmap = plt.cm.get_cmap('coolwarm')
    
    node_size = [1000 * eigenvector_centrality[node] for node in G.nodes()]  # Scale node size by centrality
    edge_color = 'gray'
    edge_width = 1.5
    alpha = 0.7 # Edge transparency
    
    # Draw graph with nodes coloured based on centrality
    nx.draw(G, pos, with_labels=False, node_size=node_size, node_color=node_color, cmap=cmap, 
            edge_color=edge_color, width=edge_width, alpha=alpha, font_weight='bold')
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min_centrality, vmax=max_centrality))
    sm.set_array([])
    plt.colorbar(sm, label="Eigenvector Centrality")
    plt.show()

adjacency_matrix = read_adjacency_matrix(filename)
G = create_adjacency_matrix_graph(adjacency_matrix)
draw_graph(G)


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
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

filename = r"GBR\tides_only\swain\connectivity_matrix.csv"

def read_adjacency_matrix(filename):
    adjacency_matrix = np.genfromtxt(filename, delimiter=',', skip_header=1)
    return adjacency_matrix

# Creating simple graph for the adjacency matrix
def create_adjacency_matrix_graph(adjacency_matrix):
    G = nx.MultiDiGraph()
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

def draw_graph(G):
    plt.figure(figsize=(10,10))
    nx.draw(G, with_labels=False, node_color="blue", edge_color="black", node_size=500)
    plt.show()

adjacency_matrix = read_adjacency_matrix(filename)
G = create_adjacency_matrix_graph(adjacency_matrix)
draw_graph(G)

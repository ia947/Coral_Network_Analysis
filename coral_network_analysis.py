# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:39:13 2024

@author: isaac
"""

#Import initial modules
import argparse
import numpy as np
import networkx as nx
import pandas as pd
#import geopandas as gpd
import matplotlib as plt
import os
import tarfile

#################################
###### DATA PRE-PROCESSING ######
#################################

## *Make sure all adjacency matrices are in .npy or .csv before reading* ##

def read_adjacency_matrix(filename):
    adjacency_matrix = np.load(filename)
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



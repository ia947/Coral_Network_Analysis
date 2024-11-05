# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:39:13 2024

@author: isaac
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

#path = "SynologyDrive/Documents/University of York/BSc (Hons) Environmental Geography/3rd Year (2024-2025)/Dissertation/Code and Data/"
#caribbean_data = np.load(path+"D_Caribbean_revised.npy")
caribbean_data = np.load("D_Caribbean_revised.npy")
#caribbean_data = pd.DataFrame(caribbean_data)

#print(caribbean_data)

G = nx.MultiDiGraph()
num_nodes = len(caribbean_data)
G.add_nodes_from(range(num_nodes))
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if caribbean_data[i][j] > 0:
            G.add_edge(i,j)
            
nx.draw(G, with_labels=False, node_color='lightblue', node_size=5, font_weight='bold')

plt.show()

adjacency_matrix = caribbean_data

def create_adjacency_matrix_graph(adjacency_matrix):
    G = nx.MultiDiGraph()
    num_nodes = len(adjacency_matrix)
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j] > 0:
                G.add_edge(i,j)
    return G

print(G)

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:39:13 2024

@author: isaac
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


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



#############################

# Function to compute centrality measures
def compute_centralities(G):
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

    return pd.DataFrame({
        'Node': list(G.nodes),
        'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
        'Closeness Centrality': [closeness_centrality[node] for node in G.nodes()],
        'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()],
        'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()],
        'Harmonic Centrality': [harmonic_centrality[node] for node in G.nodes()],
        'Clustering Coefficient': [clustering_coefficient[node] for node in G.nodes()]
    })

# Graph properties
def compute_graph_properties(G):
    density = nx.density(G)
    transitivity = nx.transitivity(G)
    # Rich club coefficient is a dictionary keyed by degree
    rich_club_coefficient = nx.rich_club_coefficient(G, normalized=False)
    
    return pd.DataFrame({
        'Property': ['Graph Density', 'Transitivity'],
        'Value': [density, transitivity]
    }), rich_club_coefficient

# Compute metrics and store results
centralities = compute_centralities(G)
graph_props, rich_club_coeff = compute_graph_properties(G)

# Output centralities to a table
print("Centrality Measures:")
print(centralities)

# Output graph properties
print("\nGraph Properties:")
print(graph_props)

# Rich Club Coefficient
print("\nRich Club Coefficient (by degree):")
print(rich_club_coeff)



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

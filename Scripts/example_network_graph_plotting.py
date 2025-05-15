# -*- coding: utf-8 -*-
"""
Created on Tue May 13 09:45:06 2025

@author: isaac
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

##############################################
###### FIGURE 2.1a - SIMPLE A-B NETWORK ######
##############################################

# Create a simple graph
G1 = nx.Graph()
G1.add_node("A")
G1.add_node("B")
G1.add_edge("A", "B")

# Draw the graph
pos = nx.spring_layout(G1)  # sets positions for all nodes
nx.draw(
    G1,
    pos,
    with_labels=True,
    node_size=700,
    node_color='lightblue',
    font_size=14,
    font_weight='bold',
    edge_color='gray'
)

# Display the plot
plt.axis('off')
plt.show()


##########################################################################
###### FIGURE 2.1b - SIMPLE A-B-C NETWORK WITH MISSING INTERACTIONS ######
##########################################################################

# Create the graph
G2 = nx.Graph()
G2.add_node("A")
G2.add_node("B")
G2.add_node("C")
G2.add_edge("A", "B")

# Draw the graph
pos = nx.spring_layout(G2)
nx.draw(
    G2,
    pos,
    with_labels=True,
    node_size=700,
    node_color='lightblue',
    font_size=14,
    font_weight='bold',
    edge_color='gray'
)

# Display the plot
plt.axis('off')
plt.show()


#################################################################
###### FIGURE 2.2 - 3-NODE NETWORK ILLUSTRATING SELF-EDGES ######
#################################################################

# Create the graph
G3 = nx.Graph()
G3.add_node("A")
G3.add_node("B")
G3.add_node("C")
G3.add_edge("A", "C")
G3.add_edge("A", "A")
G3.add_edge("B", "B")

# Draw the graph
pos = nx.spring_layout(G3)
nx.draw(
    G3,
    pos,
    with_labels=True,
    node_size=700,
    node_color='lightblue',
    font_size=14,
    font_weight='bold',
    edge_color='gray'
)

# Display the plot
plt.axis('off')
plt.show()


##################################################
###### FIGURE 2.3a - HUB-AND-SPOKE  NETWORK ######
##################################################

G4 = nx.Graph()
G4.add_node("A")
G4.add_node("B")
G4.add_node("C")
G4.add_node("D")
G4.add_node("E")
G4.add_node("F")
G4.add_edge("A", "B")
G4.add_edge("A", "C")
G4.add_edge("A", "D")
G4.add_edge("A", "E")
G4.add_edge("A", "F")

# Draw the graph
pos = nx.spring_layout(G4)
nx.draw(
    G4,
    pos,
    with_labels=True,
    node_size=700,
    node_color='lightblue',
    font_size=14,
    font_weight='bold',
    edge_color='gray'
)

# Display the plot
plt.axis('off')
plt.show()


#########################################
###### FIGURE 2.3b - MESH  NETWORK ######
#########################################

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# Define nodes
nodes = ["A", "B", "C", "D", "E", "F"]

# Create graph and add nodes
G5 = nx.Graph()
G5.add_nodes_from(nodes)

# Generate almost-complete mesh
all_pairs = list(combinations(nodes, 2))

# Specify pairs to omit
omit_edges = {("E", "F"), ("C", "D")}

# Filter edges to include all except those in omit_edges
edges = [pair for pair in all_pairs if pair not in omit_edges]
G5.add_edges_from(edges)

# Draw the mesh network
pos = nx.spring_layout(G5, seed=42)  # fixed seed for reproducible layout
nx.draw(
    G5, pos,
    with_labels=True,
    node_size=700,
    node_color='lightblue',
    font_size=14,
    font_weight='bold',
    edge_color='gray'
)

plt.axis('off')
plt.show()

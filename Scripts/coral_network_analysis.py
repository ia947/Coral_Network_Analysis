# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:39:13 2024

@author: isaac
"""


import os
import numpy as np
import networkx as nx
import graphviz
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
#from netCDF4 import Dataset


#################################
###### DATA PRE-PROCESSING ######
#################################

# Define locations for corresponding adjacency matrix filenames
locations = {
    "IO": os.path.join("IO", "IO_single_step_explicit_mean_connectivity_matrix.csv"),
    "GBR_tides-only_Cairns": os.path.join("GBR", "tides_only", "cairns", "connectivity_decimal.csv"),
    "GBR_tides-only_Grenville": os.path.join("GBR", "tides_only", "grenville", "connectivity_decimal.csv"),
    "GBR_tides-only_Swain": os.path.join("GBR", "tides_only", "swain", "connectivity_decimal.csv"),
    "GBR_wind-and-tides_Cairns": os.path.join("GBR", "wind_and_tides", "cairns", "connectivity_decimal.csv"),
    "GBR_wind-and-tides_Grenville": os.path.join("GBR", "wind_and_tides", "grenville", "connectivity_decimal.csv"),
    "GBR_wind-and-tides_Swain": os.path.join("GBR", "wind_and_tides", "swain", "connectivity_decimal.csv"),
    "GBR_wind-only_Cairns": os.path.join("GBR", "wind_only", "cairns", "connectivity_decimal.csv"),
    "GBR_wind-only_Grenville": os.path.join("GBR", "wind_only", "grenville", "connectivity_decimal.csv"),
    "GBR_wind-only_Swain": os.path.join("GBR", "wind_only", "swain", "connectivity_decimal.csv"),
    "Caribbean": os.path.join("Caribbean", "D_Caribbean_revised.npy")
    }

def read_adjacency_matrix(filename):
    if filename.endswith(".csv"):
        return np.genfromtxt(filename, delimiter=',', skip_header=0)
    elif filename.endswith(".npy"):
        return np.load(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

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
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.03, pad=0.04)
    cbar.set_label("Degree Centrality", fontsize=14)
    
    plt.show()


###############################
###### NETWORK MEASURES #######
###############################

# Function to compute all network measures
def compute_network_metrics(G, region_name):
    # Centrality measures
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    harmonic_centrality = nx.harmonic_centrality(G)
    clustering_coefficient = nx.clustering(G.to_undirected())
    
    # Graph-level measures
    density = nx.density(G)
    
    G_no_selfloops = G.copy()  # Create copy of the graph
    G_no_selfloops.remove_edges_from(nx.selfloop_edges(G_no_selfloops))  # Remove self-loops
    if G_no_selfloops.number_of_edges() > 0:  # Ensure the graph has edges
        rich_club_coefficient = nx.rich_club_coefficient(G_no_selfloops.to_undirected(), normalized=False)
        avg_rich_club = np.mean(list(rich_club_coefficient.values()))
    else:
        avg_rich_club = 0  # If no valid calculation, default to 0
    
    transitivity = nx.transitivity(G)
    local_efficiency = nx.global_efficiency(G.to_undirected())
    
    # Compute network centralisation
    degree_centrality = nx.degree_centrality(G)
    max_centrality = max(degree_centrality.values())
    N = len(G)
    network_centralisation = (sum(max_centrality - c for c in degree_centrality.values()) / ((N - 1) * (N - 2))) if N > 2 else 0
    
    # Create the DataFrame with all metrics
    metrics_df = pd.DataFrame({
        'Node': list(G.nodes),
        'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
        'Network Centralisation': [network_centralisation] * len(G.nodes()),
        'Closeness Centrality': [closeness_centrality[node] for node in G.nodes()],
        'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()],
        'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()],
        'Harmonic Centrality': [harmonic_centrality[node] for node in G.nodes()],
        'Clustering Coefficient': [clustering_coefficient[node] for node in G.nodes()],
        'Graph Density': [density] * len(G.nodes()),
        'Rich Club Coefficient': [avg_rich_club] * len(G.nodes()),
        'Transitivity': [transitivity] * len(G.nodes()),
        'Local Efficiency': [local_efficiency] * len(G.nodes())
    })
    
    # Save results to csv
    #output_filename = f"{region_name}_network_metrics.csv"
    #metrics_df.to_csv(output_filename, index=False, mode="w")
    #print(f"Metrics for {region_name} saved to {output_filename}")
    
    return metrics_df


##############################################
####### EXECUTE SCRIPT FOR ALL REGIONS #######
##############################################

for region, filename in locations.items():
    print(f"Processing {region}...")
    
    # Read and process the adjacency matrix
    adjacency_matrix = read_adjacency_matrix(filename)
    G = create_adjacency_matrix_graph(adjacency_matrix)
    
    # Visualise the graph
    draw_graph(G, use_graphviz=True)
    
    # Compute and save the network metrics
    compute_network_metrics(G, region)


###################################
###### SIMULATE NODE REMOVAL ######
###################################

def simulate_node_removal(G, metric='degree', removal_percent=10):
    """Simulate removal of top X% nodes by centrality metric and compute efficiency loss"""
    # Calculate centrality with error handling
    try:
        if metric == 'degree':
            scores = nx.degree_centrality(G)
        elif metric == 'eigenvector':
            scores = nx.eigenvector_centrality(G, max_iter=1000)
        elif metric == 'betweenness':
            scores = nx.betweenness_centrality(G, normalized=True)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    except Exception as e:
        print(f"Error calculating {metric} centrality: {str(e)}")
        return None

    # Identify top nodes
    nodes = sorted(scores, key=scores.get, reverse=True)
    if not nodes:
        print("No nodes available for removal")
        return None
    
    n_remove = max(1, int(len(nodes) * removal_percent / 100))
    nodes_to_remove = nodes[:n_remove]
    
    # Remove nodes and calculate efficiency loss
    try:
        G_removed = G.copy()
        G_removed.remove_nodes_from(nodes_to_remove)
        original_efficiency = nx.global_efficiency(G.to_undirected())
        original_edges = G.number_of_edges()
        new_efficiency = nx.global_efficiency(G_removed.to_undirected())
        edges_lost = original_edges - G_removed.number_of_edges()
    except Exception as e:
        print(f"Error during node removal: {str(e)}")
        return None
    
    return {
        'region': region,
        'metric': metric,
        'removal_percent': removal_percent,
        'edges_lost': edges_lost,
        'efficiency_loss_percent': (1 - new_efficiency/original_efficiency)*100,
        'nodes_removed': nodes_to_remove,
        'original_efficiency': original_efficiency,
        'new_efficiency': new_efficiency,
        'nodes_removed': nodes_to_remove,
        'node_impacts': {node: scores[node] for node in nodes_to_remove}
    }

# Configuration for different regions
SIMULATION_PARAMS = {
    "Caribbean": {'metric': 'eigenvector', 'removal_percent': [5, 10, 15]},
    "IO": {'metric': 'betweenness', 'removal_percent': [5, 10, 15]},
    "GBR": {'metric': 'degree', 'removal_percent': [5, 10, 15]}
}

# Modified execution loop with simulations
results = []

for region, filename in locations.items():
    print(f"\n{'='*40}\nProcessing {region}\n{'='*40}")
    
    # Read and process the adjacency matrix
    adjacency_matrix = read_adjacency_matrix(filename)
    G = create_adjacency_matrix_graph(adjacency_matrix)
    
    # Compute metrics (optional)
    metrics_df = compute_network_metrics(G, region)
    
    # Determine simulation parameters
    base_region = region.split('_')[0]
    params = SIMULATION_PARAMS.get(base_region, {'metric': 'degree', 'removal_percent': [10]})  # Default to list
    
    # Iterate over each removal percentage
    for removal_pct in params['removal_percent']:
        # Run simulation for this percentage
        sim_result = simulate_node_removal(
            G, 
            metric=params['metric'], 
            removal_percent=removal_pct  # Pass single integer
        )
        
        if sim_result:
            # Add metadata
            sim_result.update({
                'node_count': len(G.nodes()),
                'edge_count': len(G.edges()),
                'subregion': region.split('_')[-1] if 'GBR' in region else None,
                'condition': '_'.join(region.split('_')[1:-1]) if 'GBR' in region else None
            })
            results.append(sim_result)
            
            # Print summary
            print(f"{region} efficiency loss ({params['metric']} @ {removal_pct}%): "
                  f"{sim_result['efficiency_loss_percent']:.1f}%")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("node_removal_simulations.csv", index=False)
print("\nSimulation results saved to node_removal_simulations.csv")

# Generate summary statistics
if not results_df.empty:
    print("\nSimulation Summary Statistics:")
    print(results_df.groupby(['base_region', 'metric']).agg({
        'efficiency_loss_percent': ['mean', 'std'],
        'node_count': 'mean',
        'edge_count': 'mean'
    }))
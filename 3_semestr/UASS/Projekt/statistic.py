import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# Načtení statické nebo dynamické sítě
def load_network(file_path, is_temporal=False, temporal_time_col=None):
    if is_temporal:
        # Načítání dynamických (temporálních) sítí
        data = pd.read_csv(file_path)
        snapshots = {}
        for t in sorted(data[temporal_time_col].unique()):
            snapshots[t] = nx.from_pandas_edgelist(
                data[data[temporal_time_col] == t], 'source', 'target'
            )
        return snapshots  # Vrací časové snímky
    else:
        # Načítání statické sítě
        data = pd.read_csv(file_path)
        return nx.from_pandas_edgelist(data, 'source', 'target')


# Analýza vlastností statické sítě
def analyze_static_network(graph):
    degree_sequence = [d for _, d in graph.degree()]
    degree_count = Counter(degree_sequence)
    
    analysis = {
        'Nodes': graph.number_of_nodes(),
        'Edges': graph.number_of_edges(),
        'Density': nx.density(graph),
        'Average Degree': np.mean(degree_sequence),
        'Average Weighted Degree': np.mean(degree_sequence),  # Platí pro nevážené grafy
        'Average Clustering Coefficient': nx.average_clustering(graph),
        'Number of Communities': len(list(nx.community.greedy_modularity_communities(graph))),
        'Max Degree Node': max(graph.degree, key=lambda x: x[1]),
        'Degree Distribution': degree_count
    }
    return analysis


# Analýza dynamické sítě
def analyze_temporal_network(snapshots):
    analysis = {}
    for t, graph in snapshots.items():
        snapshot_analysis = analyze_static_network(graph)
        analysis[f"Snapshot {t}"] = snapshot_analysis
    return analysis


# Zajímavé metriky
def additional_metrics(graph):
    metrics = {}
    if nx.is_connected(graph):
        metrics['Diameter'] = nx.diameter(graph)
        metrics['Average Shortest Path Length'] = nx.average_shortest_path_length(graph)
    else:
        largest_component = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_component)
        metrics['Diameter (Largest Component)'] = nx.diameter(subgraph)
        metrics['Average Shortest Path Length (Largest Component)'] = nx.average_shortest_path_length(subgraph)
    
    # Rozložení stupňů
    degree_sequence = sorted([d for _, d in graph.degree()], reverse=True)
    metrics['Degree Sequence'] = degree_sequence
    return metrics


# Vizualizace rozložení stupňů
def plot_degree_distribution(graph, title):
    degree_sequence = sorted([d for _, d in graph.degree()], reverse=True)
    degree_count = Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    plt.figure(figsize=(8, 6))
    plt.bar(deg, cnt, width=0.80, color='b')
    plt.title(title)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()


# Hlavní funkce
def main():
    # Statická síť
    static_network_file = 'path_to_static_network.csv'  # Nahraďte vlastní cestou
    static_graph = load_network(static_network_file)
    static_analysis = analyze_static_network(static_graph)
    print("Static Network Analysis:")
    for key, value in static_analysis.items():
        print(f"{key}: {value}")
    additional = additional_metrics(static_graph)
    print("\nAdditional Metrics for Static Network:")
    for key, value in additional.items():
        print(f"{key}: {value}")
    plot_degree_distribution(static_graph, "Degree Distribution for Static Network")

    # Dynamická síť
    temporal_network_file = 'path_to_temporal_network.csv'  # Nahraďte vlastní cestou
    temporal_snapshots = load_network(temporal_network_file, is_temporal=True, temporal_time_col='time')
    temporal_analysis = analyze_temporal_network(temporal_snapshots)
    print("\nTemporal Network Analysis:")
    for snapshot, analysis in temporal_analysis.items():
        print(f"{snapshot}:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")

    # Rozložení stupňů pro dynamické snímky
    for t, graph in temporal_snapshots.items():
        plot_degree_distribution(graph, f"Degree Distribution for Snapshot {t}")


if __name__ == "__main__":
    main()

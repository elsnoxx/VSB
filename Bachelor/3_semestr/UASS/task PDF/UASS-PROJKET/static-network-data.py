import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Load the space-separated file into a numpy array
file_path = 'out.opsahl-usairport'
data = np.loadtxt(file_path, skiprows=1)

# Create a graph from the numpy array
G = nx.Graph()
for edge in data:
    G.add_edge(int(edge[0]), int(edge[1]))

# Remove isolated nodes
G.remove_nodes_from(list(nx.isolates(G)))

# Compute network statistics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
average_degree = sum(dict(G.degree()).values()) / num_nodes
density = nx.density(G)
is_connected = nx.is_connected(G)
diameter = nx.diameter(G) if is_connected else 'Graph is not connected'

# Print network statistics
print("Základní vlastnosti grafu:")
print(f"- Počet uzlů: {num_nodes}")
print(f"- Počet hran: {num_edges}")
print(f"- Průměrný stupeň uzlu: {average_degree:.2f}")
print(f"- Síť je souvislá: {'Ano' if is_connected else 'Ne'}")

if is_connected:
    print(f"- Průměrná délka cesty: {nx.average_shortest_path_length(G):.2f}")
    print(f"- Průměrný průměr grafu (diameter): {diameter}")
else:
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    print(f"- Velikost největší souvislé komponenty (uzly): {len(largest_cc)}")
    print(f"- Počet uzlů v souvislé komponentě: {subgraph.number_of_nodes()}")
    print(f"- Počet hran v souvislé komponentě: {subgraph.number_of_edges()}")
    print(f"- Průměrná délka cesty v souvislé komponentě: {nx.average_shortest_path_length(subgraph):.2f}")
    print(f"- Průměrný průměr grafu (diameter) v souvislé komponentě: {nx.diameter(subgraph)}")

print("\nNejdůležitější uzly podle různých metrik:")
degree_centrality = nx.degree_centrality(G)
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

print("1. Stupňová centrálnost:")
for node, value in top_degree:
    print(f"   - Uzel {node}: Centrálnost {value:.4f}")

print("\nDalší vlastnosti grafu:")
print(f"- Hustota grafu: {density:.4f}")

# Export the graph to a GraphML file for Gephi
output_file = 'graph.graphml'
nx.write_graphml(G, output_file)
print(f"\nGraph data has been exported to {output_file} for Gephi.")

# Find and visualize the largest clique, star, and edge-core
cliques = list(nx.find_cliques(G))
largest_clique = max(cliques, key=len)
print(f"\nLargest clique size: {len(largest_clique)}")

stars = [node for node, degree in G.degree() if degree == max(dict(G.degree()).values())]
print(f"Largest star size: {max(dict(G.degree()).values())}")

core_number = nx.core_number(G)
max_core = max(core_number.values())
print(f"Max core number: {max_core}")

# Visualize the largest clique
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, nodelist=largest_clique, node_color='red', label='Largest Clique')
nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if u in largest_clique and v in largest_clique], edge_color='gray')
plt.legend()
plt.title("Visualization of Largest Clique")
plt.show()

# Visualize the largest star
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, nodelist=stars, node_color='yellow', label='Largest Star')
nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if u in stars or v in stars], edge_color='gray')
plt.legend()
plt.title("Visualization of Largest Star")
plt.show()

# Visualize the highest edge-core
nodes_in_max_core = [node for node, core in core_number.items() if core == max_core]
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_max_core, node_color='green', label='Highest Edge-Core')
nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if u in nodes_in_max_core and v in nodes_in_max_core], edge_color='gray')
plt.legend()
plt.title("Visualization of Highest Edge-Core")
plt.show()
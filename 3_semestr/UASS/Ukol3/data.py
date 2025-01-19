import pandas as pd
import networkx as nx

data_file = "Email-Enron.csv"
data = pd.read_csv(data_file, sep=";", header=None, names=["source", "target"])

# KROK 2: Vytvoření vážené sítě
G = nx.Graph()

for _, row in data.iterrows():
    source, target = row["source"], row["target"]
    if G.has_edge(source, target):
        G[source][target]["weight"] += 1
    else:
        G.add_edge(source, target, weight=1)

print(f"Počet uzlů: {G.number_of_nodes()}")
print(f"Počet hran: {G.number_of_edges()}")

# KROK 3: Prořídnutí sítě
# Zvolte metodu pro prořídnutí:
# 1. Filtrace podle maximálního počtu hran (top edges)
# 2. Filtrace podle minimální váhy hran (threshold)

# Metoda 1: Prořídnutí podle top počtu hran
desired_edges = 500  # Zadejte požadovaný počet hran
sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
top_edges = sorted_edges[:desired_edges]

H = nx.Graph()
H.add_edges_from((u, v, {"weight": d["weight"]}) for u, v, d in top_edges)

# Metoda 2: Prořídnutí podle minimální váhy hran
# weight_threshold = 5  # Zadejte váhový práh
# H = nx.Graph((u, v, d) for u, v, d in G.edges(data=True) if d["weight"] > weight_threshold)

# Odstranění vrcholů bez hran
H.remove_nodes_from(list(nx.isolates(H)))

print(f"Počet uzlů po prořídnutí: {H.number_of_nodes()}")
print(f"Počet hran po prořídnutí: {H.number_of_edges()}")

# KROK 4: Export vyčištěné sítě do formátu GEXF
output_file = "cleaned_network.gexf"  # Název výstupního souboru
nx.write_gexf(H, output_file)
print(f"Vyčištěná síť byla uložena do souboru: {output_file}")

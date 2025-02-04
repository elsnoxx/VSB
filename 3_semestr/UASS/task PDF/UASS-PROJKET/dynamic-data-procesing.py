import os
import pandas as pd
import networkx as nx

class Graph:
    def __init__(self):
        self.nodes = []

    def add_point(self, point):
        self.nodes.append(point)

    def load_from_file(self, file_path):
        if not os.path.exists(file_path):
            print("File not found.")
            return

        data = pd.read_csv(file_path, header=None, names=['source', 'target', 'timestamp'])
        for _, row in data.iterrows():
            self.add_point((row['source'], row['target'], row['timestamp']))

    def get_unique_timestamps(self):
        unique_timestamps = set(point[2] for point in self.nodes)
        sorted_timestamps = sorted(unique_timestamps)
        return sorted_timestamps

    def export_cumulative_snapshots(self, output_directory, number_of_snapshots):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        sorted_nodes = sorted(self.nodes, key=lambda x: x[2])
        total_nodes = len(sorted_nodes)
        nodes_per_snapshot = total_nodes // number_of_snapshots

        cumulative_nodes = []

        for i in range(1, number_of_snapshots + 1):
            current_snapshot_limit = min(i * nodes_per_snapshot, total_nodes)
            cumulative_nodes.extend(sorted_nodes[:current_snapshot_limit])

            file_path = os.path.join(output_directory, f'cumulative_snapshot_{i}.csv')

            with open(file_path, 'w') as writer:
                writer.write("Source,Target,Weight\n")
                for node in cumulative_nodes:
                    writer.write(f"{node[0]},{node[1]},1\n")

            snapshot_df = pd.DataFrame(cumulative_nodes, columns=['source', 'target', 'timestamp'])
            analysis = analyze_snapshot(snapshot_df)
            print(f'Cumulative snapshot {i} exported to {file_path}')
            print(f'Snapshot {i} analysis: {analysis}')

def analyze_snapshot(snapshot):
    G = nx.from_pandas_edgelist(snapshot, 'source', 'target')
    if len(G) == 0:
        return {
            'avg_degree': 0,
            'avg_weighted_degree': 0,
            'num_communities': 0,
            'avg_community_size': 0,
            'max_community_size': 0
        }
    avg_degree = sum(dict(G.degree()).values()) / len(G)
    avg_weighted_degree = sum(dict(G.degree(weight='weight')).values()) / len(G)
    num_communities = nx.number_connected_components(G)
    communities = list(nx.connected_components(G))
    community_sizes = [len(c) for c in communities]
    avg_community_size = sum(community_sizes) / len(community_sizes)
    max_community_size = max(community_sizes)
    return {
        'avg_degree': avg_degree,
        'avg_weighted_degree': avg_weighted_degree,
        'num_communities': num_communities,
        'avg_community_size': avg_community_size,
        'max_community_size': max_community_size
    }

def main():
    file_path = 'contacts-prox-high-school-2013.edges'
    output_directory = 'snapshots'
    number_of_snapshots = 5

    graph = Graph()
    graph.load_from_file(file_path)
    graph.export_cumulative_snapshots(output_directory, number_of_snapshots)

if __name__ == "__main__":
    main()
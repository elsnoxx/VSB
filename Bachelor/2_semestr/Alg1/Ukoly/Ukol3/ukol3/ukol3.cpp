#include "ukol3.h"


using namespace std;

class Graph {
public:
    vector<vector<int>> adjacency_matrix;

    Graph(int size) {
        adjacency_matrix.resize(size + 1, vector<int>(size + 1, 0));
    }

    void addEdge(int v, int w) {
        adjacency_matrix[v][w] = 1;
        adjacency_matrix[w][v] = 1;
    }

    void DFS(int start, vector<bool>& visited)
    {
        //cout << start << " ";
        visited[start] = true;
        for (int i = 0; i < adjacency_matrix.size(); i++) {
            if (adjacency_matrix[start][i] == 1 && (!visited[i])) {
                DFS(i, visited);
            }
        }
    }
};

void maxVert(const string& file_name, int& vertices, int& max_node, int& min_node) {
    ifstream file(file_name);
    if (!file.is_open()) {
        cerr << "Failed to open file" << endl;
    }

    unordered_set<int> unique_vertices;


    int node1, node2;
    while (file >> node1 >> node2) {
        unique_vertices.insert(node1);
        unique_vertices.insert(node2);
        max_node = max({ max_node, node1, node2 });
        min_node = min({ min_node, node1, node2 });
    }

    file.close();

    vertices = unique_vertices.size();
}

int countComponents(Graph& graph, int max_node, int min_node) {
    vector<bool> visited(max_node + 1, false);
    int components = 0;

    for (int node = min_node; node <= max_node; ++node) {
        if (!visited[node]) {
            //cout << endl;
            graph.DFS(node, visited);
            components++;
        }
    }

    return components;
}

void printMatrix(const vector<vector<int>>& matrix) {
    for (size_t i = 1; i < matrix.size(); ++i) {
        for (size_t j = 1; j < matrix[i].size(); ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

void printResults(const string& file_name) {
    int vertices = 0, max_node = 0, min_node = 999999999;
    maxVert(file_name, vertices, max_node, min_node);

    Graph graph(max_node);

    ifstream file(file_name);
    int node1, node2, edges = 0;
    while (file >> node1 >> node2) {
        graph.addEdge(node1, node2);
        edges++;
    }

    //cout << "File: " << file_name << endl;
    cout << "Number of vertices: " << vertices << endl;
    cout << "Number of edges: " << edges << endl;
    //cout << "Adjacency matrix:" << endl;
    //printMatrix(graph.adjacency_matrix);
    int component = countComponents(graph, max_node, min_node);
    cout << "Number of connected components: " << component << endl;
    cout << endl;
}

int main() {
    const string file_name1 = "C:\\Users\\admin\\Documents\\GitHub\\Alg1\\Ukoly\\Ukol3\\ukol3\\TestData\\graph1.txt";
    printResults(file_name1);

    const string file_name2 = "C:\\Users\\admin\\Documents\\GitHub\\Alg1\\Ukoly\\Ukol3\\ukol3\\TestData\\graph2.txt";
    printResults(file_name2);

    const string file_name3 = "C:\\Users\\admin\\Documents\\GitHub\\Alg1\\Ukoly\\Ukol3\\ukol3\\TestData\\graph3.txt";
    printResults(file_name3);

    return EXIT_SUCCESS;
}
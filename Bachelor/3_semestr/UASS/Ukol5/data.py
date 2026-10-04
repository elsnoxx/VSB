import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import numpy as np

# Load the graph from the GML file
G = nx.read_gml('polbooks.gml', label='id')

def simulate_SI_model(graph, initial_infected):
    infected = set(initial_infected)
    steps = 0
    infected_counts = [len(infected)]  # Track the number of infected nodes at each step
    visualize_graph(graph, infected, steps)  # Visualize the initial state
    time.sleep(1)  # Add a delay to observe the initial state
    while len(infected) < len(graph.nodes()):
        new_infected = set()
        for node in infected:
            neighbors = set(graph.neighbors(node))
            new_infected |= neighbors - infected
        infected |= new_infected
        steps += 1
        infected_counts.append(len(infected))  # Track the number of infected nodes
        visualize_graph(graph, infected, steps)  # Visualize the graph at each step
        time.sleep(1)  # Add a delay to observe each step
    return steps, infected_counts

def simulate_cascade_model(graph, initial_infected):
    infected = set(initial_infected)
    steps = 0
    while len(infected) < len(graph.nodes()):
        new_infected = set()
        for node in infected:
            neighbors = set(graph.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in infected and random.random() < 0.5:  # 50% chance to infect
                    new_infected.add(neighbor)
        if not new_infected:
            break
        infected |= new_infected
        steps += 1
    coverage = (len(infected) / len(graph.nodes())) * 100
    return steps, coverage

def visualize_graph(graph, infected, step):
    pos = nx.spring_layout(graph)
    node_colors = ['red' if node in infected else 'lightblue' for node in graph.nodes()]
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray')
    plt.title(f"Graph at step {step}")
    plt.show()

# Run a single SI model simulation starting with node 1
initial_infected = [1]
steps_needed, infected_counts = simulate_SI_model(G, initial_infected)
print(f"SI model - Počet kroků: {steps_needed}")
print(f"Počet infikovaných uzlů v každém kroku: {infected_counts}")
average_infected_per_step = np.mean(infected_counts)
print(f"Průměrný počet infikovaných uzlů na krok: {average_infected_per_step:.1f}")

# Run multiple simulations for cascade model
num_simulations = 100
cascade_steps_list = []
cascade_coverage_list = []

for _ in range(num_simulations):
    initial_infected = [random.choice(list(G.nodes()))]
    steps_needed, coverage = simulate_cascade_model(G, initial_infected)
    cascade_steps_list.append(steps_needed)
    cascade_coverage_list.append(coverage)

average_cascade_steps = np.mean(cascade_steps_list)
average_cascade_coverage = np.mean(cascade_coverage_list)

print(f"Kaskádový model - Počet kroků: {cascade_steps_list}")
print(f"Průměrný počet kroků: {average_cascade_steps:.1f}")
print(f"Kaskádový model - Pokrytí: {cascade_coverage_list}")
print(f"Průměrné pokrytí: {average_cascade_coverage:.1f} %")
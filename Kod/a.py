import osmnx as ox
import networkx as nx
from itertools import islice
from matplotlib import pyplot as plt
import random

# Create a street network graph for a city
#G = ox.graph_from_place('New York', network_type='drive')

#save graph to file
#ox.save_graphml(G, filepath='Nowy_York_map.graphml')
def get_random_nodes(G):
    nodes = list(G.nodes)
    start, end = random.choice(nodes), random.choice(nodes)
    while start == end:
        end = random.choice(nodes)
    return start, end
# Load graph from file
G = ox.load_graphml(filepath='Nowy_York_map.graphml')

# print out number of vertices and edges
print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())
    
start, end = get_random_nodes(G)
# Generate 100 random start and end nodes
start_end_pairs = []
for _ in range(100):
    start_node, end_node = get_random_nodes(G)
    start_end_pairs.append((start_node, end_node))

# Save start and end nodes to file
with open('start_end_nodes_test1.txt', 'w') as file:
    for start_node, end_node in start_end_pairs:
        file.write(f"{start_node},{end_node}\n")
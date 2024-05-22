import random
import osmnx as ox

def get_random_nodes(G):
    nodes = list(G.nodes)
    start, end = random.choice(nodes), random.choice(nodes)
    while start == end:
        end = random.choice(nodes)
    return start, end
# Load graph from file
G = ox.load_graphml(filepath='Zalipie_map.graphml')

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
with open('Test5_start_end_nodes.txt', 'w') as file:
    for start_node, end_node in start_end_pairs:
        file.write(f"{start_node},{end_node}\n")

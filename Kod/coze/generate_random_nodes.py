import random
import osmnx as ox
from test_map import get_test_map
import os

def get_random_nodes(G):
    nodes = list(G.nodes)
    start, end = random.choice(nodes), random.choice(nodes)
    while start == end:
        end = random.choice(nodes)
    return start, end
    
def generate_start_end_nodes(G, test_number, number_of_pairs):
    start_end_pairs = []
    for _ in range(number_of_pairs):
        start_node, end_node = get_random_nodes(G)
        start_end_pairs.append((start_node, end_node))

    with open(f'{test_number}/{test_number}_start_end_nodes.txt', 'w') as file:
        for start_node, end_node in start_end_pairs:
            file.write(f"{start_node},{end_node}\n")

def get_start_end_nodes(test_number, number_of_pairs):
    start_nodes = []
    end_nodes = []
    #if file does not exist, then generate it
    if not os.path.exists(f'{test_number}/{test_number}_start_end_nodes.txt'):
        G = get_test_map(test_number)
        generate_start_end_nodes(G, test_number, number_of_pairs)
        
    #if number of lines in file is different than number of pairs, then generate it
    with open(f'{test_number}/{test_number}_start_end_nodes.txt', 'r') as file:
        lines = file.readlines()
        if len(lines) != number_of_pairs:
            G = get_test_map(test_number)
            generate_start_end_nodes(G, test_number, number_of_pairs)
    with open(f'{test_number}/{test_number}_start_end_nodes.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            start, end = line.strip().split(',')
            start_nodes.append(int(start.strip()))
            end_nodes.append(int(end.strip()))

    
    return start_nodes, end_nodes

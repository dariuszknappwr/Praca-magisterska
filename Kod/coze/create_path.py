# File: create_graph.py
import networkx as nx

def create_and_add_nodes_edges():
    G = nx.Graph()
    G.add_node(1)
    G.add_nodes_from([2, 3])
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    return G

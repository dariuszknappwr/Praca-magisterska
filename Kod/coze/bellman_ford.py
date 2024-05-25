import networkx as nx
from profiler import profile

def initialize_bellman_ford_edge_usage(G):
    #Initialize or reset 'algorithm_uses' attribute for all edges to 0.
    nx.set_edge_attributes(G, 0, 'bellman_ford_algorithm_uses')

@profile
def bellman_ford(G, source):
    # Step 1: Prepare distance and predecessor dictionaries
    distance = dict.fromkeys(G, float('infinity'))
    distance[source] = 0
    pred = {node: None for node in G}

    # Step 2: Relax edges repeatedly
    for _ in range(len(G) - 1):
        for u, v, data in G.edges(data=True):
            if distance[u] + data['length'] < distance[v]:
                distance[v] = distance[u] + data['length']
                pred[v] = u
                data['bellman_ford_algorithm_uses'] += 1

    # Step 3: Check for negative weight cycles
    for u, v, data in G.edges(data=True):
        if distance[u] + data['length'] < distance[v]:
            raise nx.NetworkXUnbounded("Graph contains a negative weight cycle.")
    return distance, pred
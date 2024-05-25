import networkx as nx
from collections import deque
from profiler import profile

def initialize_spfa_edge_usage(G):
    #Initialize or reset 'algorithm_uses' attribute for all edges to 0.
    nx.set_edge_attributes(G, 0, 'spfa_algorithm_uses')

@profile
def spfa(G, start):
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start] = 0
    queue = deque([start])
    in_queue = {node: False for node in G.nodes()}
    in_queue[start] = True

    while queue:
        u = queue.popleft()
        in_queue[u] = False
        
        # Loop through each edge that comes out of `u`
        for v, adj_data in G.adj[u].items():
            for key, data in adj_data.items():  # Here we include the key in the iteration
                weight = data['length']
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    G.edges[u, v, key]['spfa_algorithm_uses'] += 1
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

    return distances
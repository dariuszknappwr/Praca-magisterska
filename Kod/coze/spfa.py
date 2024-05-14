from graph_utils import initialize_edge_usage
from collections import deque

def spfa(G, start):
    initialize_edge_usage(G)
    
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start] = 0
    queue = deque([start])
    in_queue = {node: False for node in G.nodes()}
    in_queue[start] = True

    while queue:
        u = queue.popleft()
        in_queue[u] = False
        
        for v, adj_data in G.adj[u].items():
            for key, data in adj_data.items():
                weight = data['length']
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    G.edges[u, v, key]['algorithm_uses'] += 1
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

    return distances
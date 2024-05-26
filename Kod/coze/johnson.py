from bellman_ford import bellman_ford, initialize_bellman_ford_edge_usage
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
from profiler import profile


@profile
def johnson(G, weight='length'):
    dist = {v: 0 for v in G}
    pred = {v: [] for v in G}

    # Dodanie nowego wierzchołka 'q' z krawędziami o wadze 0 do wszystkich innych wierzchołków
    G.add_node('q')
    for v in G:
        G.add_edge('q', v, weight=0, length=0)
        
    initialize_bellman_ford_edge_usage(G)
    # Zastosowanie algorytmu Bellmana-Forda do znalezienia najkrótszych ścieżek od 'q' do wszystkich innych wierzchołków
    function_output, _, _= bellman_ford(G, 'q')
    h, _ = function_output

    # Usunięcie dodanego wierzchołka 'q'
    G.remove_node('q')

    # Przeliczenie wag krawędzi w grafie za pomocą obliczonych wartości h
    for u, v, data in G.edges(data=True):
        data[weight] = data[weight] + h[u] - h[v]

    # Initialize the distance and predecessor matrices
    dist = {n: {m: float('inf') for m in G.nodes} for n in G.nodes}
    for n in G.nodes:
        dist[n][n] = 0
    pred = {n: {m: None for m in G.nodes} for n in G.nodes}

    # Initialize the distance to all edges that are present
    for u, v, data in G.edges(data=True):
        dist[u][v] = data[weight]
        pred[u][v] = u

    # Zastosowanie algorytmu Dijkstry dla każdego wierzchołka w grafie
    for node in G.nodes():
        dist_node, pred_node = single_source_dijkstra(G, source=node, target=None, weight=weight)
        
        # Wyodrębnienie odległości, czyli dostosowanie długości najkrótszych ścieżek, aby odzwierciedlały oryginalny graf
        for target in G.nodes():
            if target in dist_node:
                dist[node][target] = dist_node[target] - h[node] + h[target]
                pred[node][target] = pred_node[target]

    return dist, pred
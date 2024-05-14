import networkx as nx

def johnsons_algorithm_simplified(G):
    distances = {}
    predecessors = {}

    # Uruchomienie algorytmu Dijkstry dla każdego węzła
    for node in G.nodes():
        dist, pred = nx.single_source_dijkstra(G, source=node, weight='length')
        distances[node] = dist
        predecessors[node] = pred

    return distances, predecessors

import networkx as nx
from profiler import profile

def initialize_bellman_ford_edge_usage(G):
    # Inicjalizacja lub zresetowanie atrybutu 'bellman_ford_algorithm_uses' dla wszystkich krawędzi na 0.
    nx.set_edge_attributes(G, 0, 'bellman_ford_algorithm_uses')

@profile
def bellman_ford(G, source, weightLabel='length'):
    # Przygotowanie słowników odległości i poprzedników
    distance = dict.fromkeys(G, float('infinity'))
    distance = {source: 0 for source in G}
    pred = {node: None for node in G}

    # Relaksacja krawędzi
    for _ in range(len(G) - 1):
        for u, v, data in G.edges(data=True):
            if distance[u] + data[weightLabel] < distance[v]:
                distance[v] = distance[u] + data[weightLabel]
                pred[v] = u
                data['bellman_ford_algorithm_uses'] += 1

    # Sprawdzenie czy istnieją ujemne cykle
    for u, v, data in G.edges(data=True):
        if distance[u] + data['length'] < distance[v]:
            raise nx.NetworkXUnbounded("Graph contains a negative weight cycle")
    return distance, pred

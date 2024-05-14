import networkx as nx

def bellman_ford(G, source):
    # przygotowanie słowników odległości i poprzedników
    distance = dict.fromkeys(G, float('infinity'))
    distance[source] = 0
    pred = {node: None for node in G}

    # wielokrotnie rozluźnianie krawędzi 
    for _ in range(len(G) - 1):
        for u, v, data in G.edges(data=True):
            if distance[u] + data['length'] < distance[v]:
                distance[v] = distance[u] + data['length']
                pred[v] = u
    # Sprawdzenie czy graf zawiera cykle o ujemnej wadze
    for u, v, data in G.edges(data=True):
        if distance[u] + data['length'] < distance[v]:
            raise nx.NetworkXUnbounded("Graph contains a negative weight cycle.")
    
    return distance, pred
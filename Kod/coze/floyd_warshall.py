from graph_utils import initialize_edge_usage

def floyd_warshall(G):
    dist = {n: {m: float('inf') for m in G.nodes} for n in G.nodes}
    for n in G.nodes:
        dist[n][n] = 0
    pred = {n: {m: None for m in G.nodes} for n in G.nodes}

    for u, v, data in G.edges(data=True):
        dist[u][v] = data['length']
        pred[u][v] = u
    
    for k in G.nodes:
        for i in G.nodes:
            for j in G.nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]

    # Sprawdzenie czy graf zawiera cykle o ujemnej wadze
    for n in G.nodes:
        if dist[n][n] < 0:
            print("Graph contains a negative-weight cycle")
            return None, None

    return dist, pred

def update_edge_usage(G, pred):
    initialize_edge_usage(G)
    
    for source in G.nodes:
        for target in G.nodes:
            # Przemierzanie najkrótszej ścieżki od celu do źródła
            while target in pred[source] and pred[source][target] is not None:
                prev = pred[source][target]
                
                if prev is not None:
                    # Zwiększanie 'algorithm_uses' poprzez bezpośredni dostęp do danych krawędzi
                    if G.is_multigraph(): 
                        # Dla multigrafów zwiększ wszystkie krawędzie między prev i target
                        for key in G[prev][target]:
                            if 'algorithm_uses' in G[prev][target][key]:
                                G[prev][target][key]['algorithm_uses'] += 1
                            else:
                                G[prev][target][key]['algorithm_uses'] = 1
                    else:
                        # Dla zwykłych grafów zwiększ tylko jedną krawędź między prev i target
                        if 'algorithm_uses' in G[prev][target]:
                            G[prev][target]['algorithm_uses'] += 1
                        else:
                            G[prev][target]['algorithm_uses'] = 1
                target = prev

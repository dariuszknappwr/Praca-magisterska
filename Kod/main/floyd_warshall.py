from profiler import profile
import tracemalloc

#@profile
def floyd_warshall(G):
    tracemalloc.start()
    # Initialize the distance and path matrices
    dist = {n: {m: float('inf') for m in G.nodes} for n in G.nodes}
    for n in G.nodes:
        dist[n][n] = 0
    pred = {n: {m: None for m in G.nodes} for n in G.nodes}

    # Initialize the distance to all edges that are present
    for u, v, data in G.edges(data=True):
        dist[u][v] = data['length']
        pred[u][v] = u
    
    for k in G.nodes:
        for i in G.nodes:
            for j in G.nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]

    # Check for negative weight cycles
    for n in G.nodes:
        if dist[n][n] < 0:
            print("Graph contains a negative-weight cycle")
            return None, None
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dist, pred, peak, 0
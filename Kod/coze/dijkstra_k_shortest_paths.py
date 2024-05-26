from dijkstra import dijkstra
import osmnx as ox
from profiler import profile

@profile
def dijkstra_k_shortest_paths(Graph, start, end, K):
    G = Graph.copy()
    paths = []
    for _ in range(K):
        # Run Dijkstra's algorithm to find the shortest path
        path = dijkstra(G, start, end)[0][0]
        if not path or path[-1] != end or path[0] != start:
            break
        paths.append(path)
        # Remove the edges of the shortest path
        for i in range(len(path) - 1):
            G.remove_edge(path[i], path[i+1])
    if(len(paths) < K):
        return None
    return paths
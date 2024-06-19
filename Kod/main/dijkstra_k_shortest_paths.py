from dijkstra import dijkstra
import osmnx as ox
from profiler import profile
import tracemalloc
#@profile
def dijkstra_k_shortest_paths(Graph, start, end, K):
    tracemalloc.start()
    G = Graph.copy()
    paths = []
    for _ in range(K):
        dijkstra_output = dijkstra(G, start, end)
        path = dijkstra_output[0][0]
        if type(path) != list:
            path = [path]
        if not path or path[-1] != end or path[0] != start:
            break
        paths.append(path)
        # Usuwanie z grafu krawędzi ze znalezionego rozwiązania
        for i in range(len(path) - 1):
            G.remove_edge(path[i], path[i+1])
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if(len(paths) < K):
        return None, peak,  0
    return paths, peak, 0
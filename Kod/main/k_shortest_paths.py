from itertools import islice
import networkx as nx
from profiler import profile
import tracemalloc


#@profile
def k_shortest_paths(G, source, target, k, weight=None):
    tracemalloc.start()
    G_directed = nx.DiGraph(G)
    result = list(islice(nx.shortest_simple_paths(G_directed, source, target, weight=weight), k))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak, 0
    
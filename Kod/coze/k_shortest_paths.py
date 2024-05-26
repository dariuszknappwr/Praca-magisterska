from itertools import islice
import networkx as nx
from profiler import profile

@profile
def k_shortest_paths(G, source, target, k, weight=None):
    G_directed = nx.DiGraph(G)
    return list(
        islice(nx.shortest_simple_paths(G_directed, source, target, weight=weight), k)
    )
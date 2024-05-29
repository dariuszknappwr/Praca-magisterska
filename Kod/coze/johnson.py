from bellman_ford import bellman_ford, initialize_bellman_ford_edge_usage
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
from profiler import profile

@profile
def johnson(G, weightLabel='length'):
    return nx.johnson(G, weight=weightLabel)
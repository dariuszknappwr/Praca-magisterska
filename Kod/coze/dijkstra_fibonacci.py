from collections import deque
from heapdict import heapdict
from profiler import profile

@profile
def dijkstra_fibonacci(G, orig, dest, weightLabel='length', plot=False):

    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0

    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = heapdict()
    pq[orig] = 0
    step = 0
    while pq:
        node = pq.popitem()[0]
        if node == dest:
            break
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node):
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)][weightLabel]

            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                pq[neighbor] = G.nodes[neighbor]["distance"]
        step += 1

    if G.nodes[dest]["previous"] is None:
        return None, step

    path = deque()
    current_node = dest
    while current_node is not None:
        path.appendleft(current_node)
        current_node = G.nodes[current_node]["previous"]
    return list(path), step
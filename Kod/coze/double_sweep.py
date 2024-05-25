from dijkstra import dijkstra_end_node
import random

def double_sweep(G, start=None):
    if start is None:
        start = random.choice(list(G.nodes))

    # First sweep to find the furthest node from the start
    _, path_to_furthest = dijkstra_end_node(G, start)
    # If no path found in first sweep, terminate
    if not path_to_furthest:
        return None

    # Second sweep from the furthest node found in the first sweep
    end_of_path, path = dijkstra_end_node(G, path_to_furthest[-1])

    return path
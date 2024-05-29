from heapq import heappop, heappush
from profiler import profile

@profile
def hoffman_pavley(G, source, target, k, weightLabel='length'):
    # Initialize a list to store the k shortest paths
    k_shortest_paths = []

    # Initialize a heap to store the paths to be explored
    paths = [(0, [source])]

    while paths and len(k_shortest_paths) < k:
        # Pop the path with the minimum cost from the heap
        cost, path = heappop(paths)

        # Get the last node in the path
        node = path[-1]

        if node == target:
            # If the target node is reached, add the path to the list of k shortest paths
            k_shortest_paths.append((cost, path))
        else:
            # Explore the neighbors of the current node
            for neighbor in G.neighbors(node):
                if neighbor not in path:
                    # Calculate the cost of the new path
                    new_cost = cost + G[node][neighbor][0][weightLabel]

                    # Create a new path by extending the current path
                    new_path = path + [neighbor]

                    # Add the new path to the heap
                    heappush(paths, (new_cost, new_path))

    return k_shortest_paths
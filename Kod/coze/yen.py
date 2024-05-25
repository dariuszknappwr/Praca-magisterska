import networkx as nx

def yen_ksp(G, source, target, K=1):
    A = [nx.shortest_path(G, source, target, weight='length')]
    B = []

    for k in range(1, K):
        for i in range(len(A[k - 1]) - 1):
            # Initialize the spur node and root path
            spur_node = A[k - 1][i]
            root_path = A[k - 1][:i + 1]

            # Keep track of removed edges to restore them later
            edges_removed = []

            # Remove the edges involved in the shortest paths found so far
            for path in A:
                if len(path) > i and root_path == path[:i + 1]:
                    u, v = path[i], path[i + 1]
                    edge_keys = list(G[u][v].keys())  # Make a separate list of edge keys
                    for key in edge_keys:
                        edge_data = G[u][v][key]
                        G.remove_edge(u, v, key)
                        edges_removed.append((u, v, key, edge_data))

            spur_path = None
            try:
                # Compute a spur path from the spur node to the target
                spur_path = nx.shortest_path(G, spur_node, target, weight='length')
            except nx.NetworkXNoPath:
                pass

            # Total path is the concatenation of root path and spur path
            if spur_path is not None:
                total_path = root_path[:-1] + spur_path
                B.append(total_path)

            # Restore the edges that were removed
            for u, v, key, data in edges_removed:
                G.add_edge(u, v, key=key, **data)

        if not B:
            break  # If no spur paths are found, stop searching

        # Sort the potential k-shortest paths by their lengths
        B.sort(key=lambda path: sum(G[path[j]][path[j + 1]][0]['length'] for j in range(len(path) - 1)))
        # Add the shortest path among B to the list of k-shortest paths
        A.append(B[0])
        # Remove the shortest path added to A from list B
        B.pop(0)

    return A
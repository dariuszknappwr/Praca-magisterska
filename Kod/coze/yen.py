import queue
import networkx as nx
from graph_utils import analyze_path
from dijkstra import dijkstra
import osmnx as ox

def yen_ksp(G, source, target, K=1):
    nx.shortest_paths()
    Original_G = G.copy()
    A = []
    B = []
    path = dijkstra(G, source, target)[0][0]
    A.append(path)

    for k in range(1, K):
        for i in range(0, len(A[k-1]) - 2):
            spur_node = A[k-1][i]
            root_path = A[k-1][:i]

            for path in A:
                if root_path == path[:i] and G.has_edge(path[i], path[i + 1]):
                    G.remove_edge(path[i], path[i + 1])
            
            for node in root_path:
                if node != spur_node and G.has_node(node):
                    G.remove_node(node)
            
            spur_path = dijkstra(G, spur_node, target)[0][0]
            if not spur_path:
                G = Original_G.copy()
                continue
            total_path = root_path + spur_path

            if total_path not in B:
                B.append(total_path)
            
            G = Original_G.copy()

        if not B:
            break

        B.sort(key=lambda x: analyze_path(G, x)[0])
        A.append(B[0])
        B.pop(0)

    return A

def yen_ksp(G, source, target, weight=None):
    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")

    if target not in G:
        raise nx.NodeNotFound(f"target node {target} not in graph")

    listA = []
    listB = queue()
    prev_path = None
    while True:
        if not prev_path:
            path = dijkstra(G, source, target, weight=weight)[0][0]
            _,length,_,_ = analyze_path(G, path)
            listB.push(length, path)
        else:
            ignore_nodes = set()
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                _,root_length,_,_ = analyze_path(G, root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    length, spur = dijkstra(
                        G,
                        root[-1],
                        target,
                        ignore_nodes=ignore_nodes,
                        ignore_edges=ignore_edges,
                        weight=weight,
                    )
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except nx.NetworkXNoPath:
                    pass
                ignore_nodes.add(root[-1])

        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
        else:
            break

import heapq
import networkx as nx
from graph_utils import analyze_path
from dijkstra import dijkstra

def yen_ksp(G, source, target, K=1):
    A = []
    B = []
    path = dijkstra(G, source, target)[0][0]
    A.append(path)
    if path[-1] != target or path[0] != source:
        return None

    try:
        for k in range(1, K):
            for i in range(0, len(A[k-1]) - 2):
                spur_node = A[k-1][i]
                root_path = A[k-1][:i]

                for path in A:
                    if root_path == path[:i]:
                        G[path[i]][path[i + 1]][0]['ignored'] = True
                
                for node in root_path:
                    if node != spur_node:
                        G.nodes[node]['ignored'] = True
                
                spur_path = dijkstra(G, spur_node, target)[0][0]
                if not spur_path or spur_path[-1] != target or spur_path[0] != source:
                    return None

                total_path = root_path + spur_path

                if total_path not in B:
                    B.append(total_path)
                
                for node in root_path:
                    G.nodes[node]['ignored'] = False

                for path in A:
                    if root_path == path[:i]:
                        G[path[i]][path[i + 1]][0]['ignored'] = False
                
            if not B:
                break

            B.sort(key=lambda x: analyze_path(G, x)[0])
            A.append(B[0])
            B.pop(0)

    except:
        return None

    if len(A) < K:
        return None

    return A

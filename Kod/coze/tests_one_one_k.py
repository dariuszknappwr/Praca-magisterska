from yen import yen_ksp
from dijkstra_k_shortest_paths import dijkstra_k_shortest_paths
from graph_utils import analyze_path
from plot_graph import plot_graph
import osmnx as ox

def tests_one_one_k(G, start_nodes, end_nodes, plot=False):
    for G_node in G.nodes:
        G.nodes[G_node]['ignored'] = False
    for G_edge in G.edges:
        G.edges[G_edge]['ignored'] = False
    results = {}
    for i in range(len(start_nodes)):
        start = start_nodes[i]
        end = end_nodes[i]
        for K in [1, 2, 3, 4, 5, 10, 25, 50, 100]:
            ksp_paths = yen_ksp(G, start, end, K)
            if ksp_paths is None:
                continue
            ksp_results = []
            i = 1
            for path in ksp_paths:
                ksp_result = analyze_path(G, path)
                ksp_results.append({
                    "Path Number": i,
                    "Travel Time": ksp_result[0],
                    "Path Length": ksp_result[1],
                    "Default Speed Distance": ksp_result[2],
                    "Average Speed": ksp_result[3],
                    "Path": path
                })
                if plot:
                    ox.plot_graph_route(G, path, route_linewidth=3, node_size=0, bgcolor='k')
                i += 1

            results[f"Yen's KSP Algorithm - K={K}"] = ksp_results
            
            double_sweep_paths = dijkstra_k_shortest_paths(G, start, end, K)
            if double_sweep_paths:
                double_sweep_results = []
                i = 1
                for path in double_sweep_paths:
                    double_sweep_result = analyze_path(G, path)
                    double_sweep_results.append({
                        "Path Number": i,
                        "Travel Time": double_sweep_result[0],
                        "Path Length": double_sweep_result[1],
                        "Default Speed Distance": double_sweep_result[2],
                        "Average Speed": double_sweep_result[3],
                        "Path": path
                    })
                    if plot:
                        plot_graph(G, path, f'Double Sweep Route - K={K}')
                    i += 1

                results[f"Double Sweep Algorithm - K={K}"] = double_sweep_results
            
    return results
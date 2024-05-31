from k_shortest_paths import k_shortest_paths
from dijkstra_k_shortest_paths import dijkstra_k_shortest_paths
from hoffman_pavley import hoffman_pavley
from graph_utils import analyze_path
from plot_graph import plot_graph
import osmnx as ox
import time

def tests_one_one_k(G, start, end, plot=False):
    for G_node in G.nodes:
        G.nodes[G_node]['ignored'] = False
    for G_edge in G.edges:
        G.edges[G_edge]['ignored'] = False
    results = {}
    for K in [1, 2, 3, 4, 5, 10, 25, 50, 100]:
        start_time = time.time()
        ksp_paths, consumed_memory, consumed_cpu = k_shortest_paths(G, start, end, K, weight='length')
        end_time = time.time()
        print(f"KSP: {end_time - start_time}")
        yen_time = end_time - start_time
        if ksp_paths is not None:
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
                    "Time": yen_time,
                    "Memory": consumed_memory,
                    "CPU": consumed_cpu,
                    "Path": path
                })
                if plot:
                    ox.plot_graph_route(G, path, route_linewidth=3, node_size=0, bgcolor='k')
                i += 1

            results[f"Yen's KSP Algorithm - K={K}"] = ksp_results
            results[f"Yen's time - K={K}"] = yen_time
            results[f"Yen's memory - K={K}"] = consumed_memory
            results[f"Yen's CPU - K={K}"] = consumed_cpu

        start_time = time.time()
        dijkstra_paths, consumed_memory, consumed_cpu = dijkstra_k_shortest_paths(G, start, end, K)
        end_time = time.time()
        dijkstra_time = end_time - start_time
        print(f"Dijkstra: {end_time - start_time}")
        if dijkstra_paths:
            dijkstra_results = []
            i = 1
            for path in dijkstra_paths:
                dijkstra_result = analyze_path(G, path)
                dijkstra_results.append({
                    "Path Number": i,
                    "Travel Time": dijkstra_result[0],
                    "Path Length": dijkstra_result[1],
                    "Default Speed Distance": dijkstra_result[2],
                    "Average Speed": dijkstra_result[3],
                    "Path": path
                })
                if plot:
                    plot_graph(G, path, f'Dijkstra Route - K={K}')
                i += 1

            results[f"Dijkstra Algorithm - K={K}"] = dijkstra_results
            results[f"Dijkstra time - K={K}"] = dijkstra_time
            results[f"Dijkstra memory - K={K}"] = consumed_memory
            results[f"Dijkstra CPU - K={K}"] = consumed_cpu
        ##############
        start_time = time.time()
        hoffman_pavley_paths, consumed_memory, consumed_cpu = hoffman_pavley(G, start, end, K)
        end_time = time.time()
        hoffman_pavley_time = end_time - start_time
        print(f"Hoffman-Pavley: {end_time - start_time}")
        if hoffman_pavley_paths:
            hoffman_pavley_results = []
            i = 1
            for path in hoffman_pavley_paths:
                hoffman_pavley_result = analyze_path(G, path)
                hoffman_pavley_results.append({
                    "Path Number": i,
                    "Travel Time": hoffman_pavley_result[0],
                    "Path Length": hoffman_pavley_result[1],
                    "Default Speed Distance": hoffman_pavley_result[2],
                    "Average Speed": hoffman_pavley_result[3],
                    "Path": path
                })
                if plot:
                    plot_graph(G, path, f'Hoffman-Pavley Route - K={K}')
                i += 1

            results[f"Hoffman-Pavley Algorithm - K={K}"] = hoffman_pavley_results
            results[f"Hoffman-Pavley time - K={K}"] = hoffman_pavley_time
            results[f"Hoffman-Pavley memory - K={K}"] = consumed_memory
            results[f"Hoffman-Pavley CPU - K={K}"] = consumed_cpu
        
    return results
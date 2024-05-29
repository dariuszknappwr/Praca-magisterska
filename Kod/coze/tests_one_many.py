from bellman_ford import bellman_ford, initialize_bellman_ford_edge_usage
from spfa import spfa, initialize_spfa_edge_usage
from plot_graph import plot_heatmap
import time
import matplotlib.colors as mcolors


def tests_one_many(G, start_node, end_node, plot=False):
    initialize_spfa_edge_usage(G)
    initialize_bellman_ford_edge_usage(G)

    start_time = time.time()
    function_output, bellman_ford_consumed_memory, bellamn_ford_consumed_cpu = bellman_ford(G, start_node)
    bellman_ford_distances, pred = function_output
    end_time = time.time()
    bellman_ford_algorithm_time = end_time - start_time

    distSum = 0
    bellman_ford_finite_length_paths_count = 0
    if bellman_ford_distances[end_node] != float('inf'):
        distSum += bellman_ford_distances[end_node]
        bellman_ford_finite_length_paths_count += 1
    if bellman_ford_finite_length_paths_count > 0:
        bellman_ford_dist_average = distSum / bellman_ford_finite_length_paths_count
    else:
        bellman_ford_dist_average = float('inf')

    start_time = time.time()
    spfa_distances, spfa_consumed_memory, spfa_consumed_cpu = spfa(G, start_node)
    end_time = time.time()
    spfa_algorithm_time = end_time - start_time

    distSum = 0
    spfa_finite_length_paths_count = 0
    if spfa_distances[end_node] != float('inf'):
        distSum += spfa_distances[end_node]
        spfa_finite_length_paths_count += 1
    if spfa_finite_length_paths_count > 0:
        spfa_dist_average = distSum / spfa_finite_length_paths_count
    else:
        spfa_dist_average = float('inf')

    result = {}
    result.update({
        f"Bellman Ford Time": bellman_ford_algorithm_time,
        f"Bellman Ford Average Distance": bellman_ford_dist_average,
        f"Bellman Ford Finite Length Paths Count": bellman_ford_finite_length_paths_count,
        f"Bellman Ford Consumed Memory": bellman_ford_consumed_memory,
        f"Bellman Ford Consumed CPU": bellamn_ford_consumed_cpu,
        f"Bellman Ford Distances": bellman_ford_distances,
        f"SPFA Time": spfa_algorithm_time,
        f"SPFA Average Distance": spfa_dist_average,
        f"SPFA Finite Length Paths Count": spfa_finite_length_paths_count,
        f"SPFA Consumed Memory": spfa_consumed_memory,
        f"SPFA Consumed CPU": spfa_consumed_cpu,
        f"SPFA Distances": spfa_distances
        })
    
    spfa_sum = 0
    bellman_ford_sum = 0
    for u,v,data in G.edges(data=True):
        spfa_sum += data['spfa_algorithm_uses']
        bellman_ford_sum += data['bellman_ford_algorithm_uses']
    if plot:
        plot_heatmap(G, 'spfa_algorithm_uses')
        plot_heatmap(G, 'bellman_ford_algorithm_uses')

    return result
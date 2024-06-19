from bellman_ford import bellman_ford, initialize_bellman_ford_edge_usage
from spfa import spfa, initialize_spfa_edge_usage
from plot_graph import plot_heatmap
import time
import matplotlib.colors as mcolors


def tests_one_many(G, start_node, end_node, plot=False):
    initialize_spfa_edge_usage(G)
    initialize_bellman_ford_edge_usage(G)

    start_time = time.time()
    bellman_ford_distances, pred, bellman_ford_consumed_memory, bellamn_ford_consumed_cpu = bellman_ford(G, start_node)
    end_time = time.time()
    bellman_ford_algorithm_time = end_time - start_time

    bellman_ford_distSum = 0
    bellman_ford_finite_length_paths_count = 0
    for dist in bellman_ford_distances.values():
        if dist != float('inf'):
            bellman_ford_distSum += dist
            bellman_ford_finite_length_paths_count += 1
    if bellman_ford_finite_length_paths_count > 0:
        bellman_ford_dist_average = bellman_ford_distSum / bellman_ford_finite_length_paths_count
    else:
        bellman_ford_dist_average = float('inf')

    start_time = time.time()
    spfa_distances, spfa_consumed_memory, spfa_consumed_cpu = spfa(G, start_node)
    end_time = time.time()
    spfa_algorithm_time = end_time - start_time

    spfa_distSum = 0
    spfa_finite_length_paths_count = 0
    for dist in spfa_distances.values():
        if dist != float('inf'):
            spfa_distSum += dist
            spfa_finite_length_paths_count += 1
    if spfa_finite_length_paths_count > 0:
        spfa_dist_average = spfa_distSum / spfa_finite_length_paths_count
    else:
        spfa_dist_average = float('inf')

    result = {}
    result.update({
        f"Bellman Ford Time": bellman_ford_algorithm_time,
        f"Bellman Ford Distance Sum": bellman_ford_distSum,
        f"Bellman Ford Average Distance": bellman_ford_dist_average,
        f"Bellman Ford Finite Length Paths Count": bellman_ford_finite_length_paths_count,
        f"Bellman Ford Consumed Memory": bellman_ford_consumed_memory,
        f"Bellman Ford Consumed CPU": bellamn_ford_consumed_cpu,
        #f"Bellman Ford Distances": bellman_ford_distances,
        f"SPFA Time": spfa_algorithm_time,
        f"SPFA Distance Sum": spfa_distSum,
        f"SPFA Average Distance": spfa_dist_average,
        f"SPFA Finite Length Paths Count": spfa_finite_length_paths_count,
        f"SPFA Consumed Memory": spfa_consumed_memory,
        f"SPFA Consumed CPU": spfa_consumed_cpu,
        #f"SPFA Distances": spfa_distances
        f"Results match": bellman_ford_distances == spfa_distances
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
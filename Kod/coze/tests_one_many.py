from bellman_ford import bellman_ford, initialize_bellman_ford_edge_usage
from spfa import spfa, initialize_spfa_edge_usage
from plot_graph import plot_heatmap
import time
import matplotlib.colors as mcolors


def tests_one_many(G, start_nodes, end_nodes, plot=False):
    initialize_spfa_edge_usage(G)
    initialize_bellman_ford_edge_usage(G)
    for i in range(len(start_nodes)):
        start_node = start_nodes[i]
        end_node = end_nodes[i]

        start_time = time.time()
        function_output, bellman_ford_consumed_memory, bellamn_ford_consumed_cpu = bellman_ford(G, start_node)
        distances, pred = function_output
        end_time = time.time()
        bellman_ford_algorithm_time = end_time - start_time

        distSum = 0
        bellman_ford_finite_length_paths_count = 0
        for end in end_nodes:
            if distances[end] != float('inf'):
                distSum += distances[end]
                bellman_ford_finite_length_paths_count += 1
        if bellman_ford_finite_length_paths_count > 0:
            bellman_ford_dist_average = distSum / bellman_ford_finite_length_paths_count
        else:
            bellman_ford_dist_average = float('inf')

        start_time = time.time()
        distances, spfa_consumed_memory, spfa_consumed_cpu = spfa(G, start_node)
        end_time = time.time()
        spfa_algorithm_time = end_time - start_time

        distSum = 0
        spfa_finite_length_paths_count = 0
        for end in end_nodes:
            if distances[end] != float('inf'):
                distSum += distances[end]
                spfa_finite_length_paths_count += 1
        if spfa_finite_length_paths_count > 0:
            spfa_dist_average = distSum / spfa_finite_length_paths_count
        else:
            spfa_dist_average = float('inf')

        result = {}
        if distances:
            result.update({
                f"Bellman Ford Time": bellman_ford_algorithm_time,
                f"Bellman Ford Average Distance": bellman_ford_dist_average,
                f"Bellman Ford Finite Length Paths Count": bellman_ford_finite_length_paths_count,
                f"Bellman Ford Consumed Memory": bellman_ford_consumed_memory,
                f"Bellman Ford Consumed CPU": bellamn_ford_consumed_cpu,
                f"SPFA Time": spfa_algorithm_time,
                f"SPFA Average Distance": spfa_dist_average,
                f"SPFA Finite Length Paths Count": spfa_finite_length_paths_count,
                f"SPFA Consumed Memory": spfa_consumed_memory,
                f"SPFA Consumed CPU": spfa_consumed_cpu
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
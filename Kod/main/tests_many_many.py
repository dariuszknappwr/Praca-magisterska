from matplotlib import pyplot as plt
from floyd_warshall import floyd_warshall
from johnson import johnson
import osmnx as ox
import networkx as nx
import time
from plot_graph import plot_heatmap
from graph_utils import initialize_edge_usage, update_edge_usage, update_edge_usage_johnson
import os
from hoffman_pavley import hoffman_pavley


def tests_many_many(G, test_number, iteration_number, plot=False):
    # Ensure the Graph has a pos attribute for plotting
    G = ox.project_graph(G)
    #G_all_algorithm_uses = G.copy()

    floyd_warshall_iterations = 0
    johnson_iterations = 0

    # Initialize 'algorithm_uses' for all edges to 0
    initialize_edge_usage(G)
    #initialize_edge_usage(G_all_algorithm_uses)
    # Run Floyd-Warshall algorithm
    start_time = time.time()
    floyd_warshall_distances, floyd_warshall_predecessors, floyd_warshall_consumed_memory, floyd_warshall_consumed_cpu = floyd_warshall(G)
    end_time = time.time()
    floyd_warshall_algorithm_time = end_time - start_time
        
    
    
    # Update edge usage based on the Floyd-Warshall algorithm
    if floyd_warshall_distances and floyd_warshall_predecessors:
        update_edge_usage(G, floyd_warshall_predecessors)
        for u,v,data in G.edges(data=True):
            floyd_warshall_iterations += data['algorithm_uses']
        # Plot heatmap
        if plot:
            plot_heatmap(G, 'Iteracje algorytmu')
            os.makedirs(f'{test_number}/floyd_warshall', exist_ok=True)
            plt.savefig(f'{test_number}/floyd_warshall/heatmap_floyd_warshall_test_{test_number}_iteration_{iteration_number}.png')
            plt.close()

    
    # To be placed within the `main` function, replacing the previous heatmap plotting section

    # Initialize 'length' for all edges to zero
    initialize_edge_usage(G)
    start_time = time.time()
    johnson_distances, johnson_consumed_memory, johnson_consumed_cpu = johnson(G)
    end_time = time.time()
    johnsons_algorithm_time = end_time - start_time



    if johnson_distances:
        update_edge_usage_johnson(G, johnson_distances)
        for u,v,data in G.edges(data=True):
            johnson_iterations += data['algorithm_uses']
        # Plot heatmap
        if plot:
            plot_heatmap(G, 'algorithm_uses')
            os.makedirs(f'{test_number}/johnson', exist_ok=True)
            plt.savefig(f'{test_number}/johnson/heatmap_johnson_test_{test_number}_iteration_{iteration_number}.png')
            plt.close()

    initialize_edge_usage(G)

    result = {
        "Floyd-Warshall Time": floyd_warshall_algorithm_time,
        "Floyd-Warshall Iterations": floyd_warshall_iterations,
        "Floyd-Warshall Consumed Memory": floyd_warshall_consumed_memory,
        "Floyd-Warshall Consumed CPU": floyd_warshall_consumed_cpu,
        "Johnson's Time": johnsons_algorithm_time,
        "Johnson's Iterations": johnson_iterations,
        "Johnson's Algorithm Consumed Memory": johnson_consumed_memory,
        "Johnson's Algorithm Consumed CPU": johnson_consumed_cpu
    }

    return result
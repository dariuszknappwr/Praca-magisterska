from floyd_warshall import floyd_warshall
from johnson import johnson
import osmnx as ox
import networkx as nx
import time
from plot_graph import plot_heatmap
from graph_utils import initialize_edge_usage, update_edge_usage



def tests_many_many(G, plot=False):
    # Ensure the Graph has a pos attribute for plotting
    G = ox.project_graph(G)

    # Initialize 'algorithm_uses' for all edges to 0
    initialize_edge_usage(G)
    # Run Floyd-Warshall algorithm
    start_time = time.time()
    function_output, floyd_warshall_consumed_memory, floyd_warshall_consumed_cpu = floyd_warshall(G)
    dist, pred = function_output
    end_time = time.time()
    floyd_warshall_algorithm_time = end_time - start_time
    
    # Update edge usage based on the Floyd-Warshall algorithm
    if dist and pred:
        update_edge_usage(G, pred)
        # Plot heatmap
        if plot:
            plot_heatmap(G, 'algorithm_uses')

    
    # To be placed within the `main` function, replacing the previous heatmap plotting section

    # Initialize 'length' for all edges to zero
    initialize_edge_usage(G)

    start_time = time.time()
    function_output, johnson_consumed_memory, johnson_consumed_cpu = johnson(G)
    distances, predecessors = function_output
    end_time = time.time()
    johnsons_algorithm_time = end_time - start_time

    if distances and predecessors:
        update_edge_usage(G, pred)
        # Plot heatmap
        if plot:
            plot_heatmap(G, 'algorithm_uses')

    floyd_warshall_sum = 0
    johnsons_algorithm_sum = 0
    for u,v,data in G.edges(data=True):
        floyd_warshall_sum += data['algorithm_uses']
        johnsons_algorithm_sum += data['algorithm_uses']

    result = {
        "Floyd-Warshall Time": floyd_warshall_algorithm_time,
        "Floyd-Warshall Iterations": floyd_warshall_sum,
        "Floyd-Warshall Consumed Memory": floyd_warshall_consumed_memory,
        "Floyd-Warshall Consumed CPU": floyd_warshall_consumed_cpu,
        "Johnson's Algorithm Consumed Memory": johnson_consumed_memory,
        "Johnson's Algorithm Time": johnsons_algorithm_time,
        "Johnson's Algorithm Iterations": johnsons_algorithm_sum,
        "Johnson's Algorithm Consumed CPU": johnson_consumed_cpu
    }

    return result
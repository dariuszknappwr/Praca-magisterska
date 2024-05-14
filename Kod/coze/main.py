from graph_creation import download_map, load_local_map
from dijkstra import dijkstra
from yen_ksp import yen_ksp
from Kod.coze.double_sweep import double_sweep_ksp
from a_star import a_star_algorithm
from bellman_ford import bellman_ford
from spfa import spfa
from floyd_warshall import floyd_warshall
from johnsons import johnsons_algorithm_simplified
from graph_utils import initialize_edge_usage
from graph_utils import update_edge_usage
from graph_analysis import analyze_path
from a_star import euclidean_heuristic
from a_star import manhattan_heuristic
from a_star import haversine
from graph_visualization import plot_graph, plot_heatmap
from graph_utils import get_random_nodes
import osmnx as ox


def main():
    # Specify the path to your local .osm file
    local_osm_file_path = 'Wroclaw_map.graphml'
    try:
        G = load_local_map(local_osm_file_path)
    except FileNotFoundError as e:
        print(e)
        return
    
    start, end = get_random_nodes(G)


    dijkstra_path = dijkstra(G, start, end)
    if dijkstra_path:
        travel_time, path_length, default_speed_distance, average_speed = analyze_path(G, dijkstra_path)
        print(f"Dijkstra's Algorithm Travel Time: {travel_time} seconds")
        print(f"Dijkstra's Algorithm Path Length: {path_length} meters")
        print(f"Dijkstra's Algorithm Default Speed Distance: {default_speed_distance} meters")
        print(f"Dijkstra's Algorithm Average Speed: {average_speed} km/h")
        plot_graph(G, dijkstra_path, 'Dijkstra Route')
    
    # Using Yen's KSP Algorithm
    ksp_paths = yen_ksp(G, start, end, K=2)
    for i, path in enumerate(ksp_paths, start=1):
        ksp_results = analyze_path(G, path)
        print(f"Yen's KSP Algorithm - Path {i} Results:")
        print(f"Travel Time: {ksp_results[0]} seconds")
        print(f"Path Length: {ksp_results[1]} meters")
        print(f"Default Speed Distance: {ksp_results[2]} meters")
        print(f"Average Speed: {ksp_results[3]} km/h")
        plot_graph(G, path, f"Yen's KSP Route {i}")

       
    ksp_paths = double_sweep_ksp(G, start, end, K=2)
    for i, path in enumerate(ksp_paths, start=1):
        ksp_results = analyze_path(G, path)
        print(f"Double Sweep KSP Algorithm - Path {i} Results:")
        print(f"Travel Time: {ksp_results[0]} seconds")
        print(f"Path Length: {ksp_results[1]} meters")
        print(f"Default Speed Distance: {ksp_results[2]} meters")
        print(f"Average Speed: {ksp_results[3]} km/h")
        plot_graph(G, path, f"Double Sweep KSP Route {i}")

    a_star_path = a_star_algorithm(G, start, end, euclidean_heuristic)
    if a_star_path:
        a_star_time, a_star_path_length, a_star_default_speed_distance, a_star_average_speed = analyze_path(G, a_star_path)
        print("A* Algorithm Results:")
        print(f"Travel Time: {a_star_time} seconds")
        print(f"Path Length: {a_star_path_length} meters")
        print(f"Default Speed Distance: {a_star_default_speed_distance} meters")
        print(f"Average Speed: {a_star_average_speed} km/h")
        plot_graph(G, a_star_path, 'A* Route')


    distances = bellman_ford(G, start)
    if distances:
        plot_heatmap(G, 'algorithm_uses')

    distances = spfa(G, start)
    if distances:
        plot_heatmap(G, 'algorithm_uses')
   
    # Ensure the Graph has a pos attribute for plotting
    G = ox.project_graph(G)

    # Initialize 'algorithm_uses' for all edges to 0
    initialize_edge_usage(G)
    # Run Floyd-Warshall algorithm
    dist, pred = floyd_warshall(G)
    
    # Update edge usage based on the Floyd-Warshall algorithm
    if dist and pred:
        update_edge_usage(G, pred)
        # Plot heatmap
        plot_heatmap(G, 'algorithm_uses')

    
    # Initialize 'length' for all edges to zero
    initialize_edge_usage(G)

    distances, predecessors = johnsons_algorithm_simplified(G)

    if distances and predecessors:
        update_edge_usage(G, pred)
        # Plot heatmap
        plot_heatmap(G, 'algorithm_uses')


if __name__ == '__main__':
    main()
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import random
import os
import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.colors as mcolors
import time
from pymongo import MongoClient
from bellman_ford import bellman_ford
from spfa import spfa
from spfa import initialize_spfa_edge_usage
from bellman_ford import initialize_bellman_ford_edge_usage
import psutil
from floyd_warshall import floyd_warshall
from johnson import johnson
from test_map import get_test_map
from generate_random_nodes import get_start_end_nodes
from tests_one_one import tests_one_one



# Define a function to calculate time from distance and speed
def calculate_time(distance, max_speed):
    speed_m_per_s = max_speed / 3.6
    time = distance / speed_m_per_s
    return time

def download_map(city, osm_file_path):
    G = ox.graph_from_place(city, network_type='drive')
    ox.save_graphml(G, filepath=osm_file_path)
    return G


def get_random_nodes(G):
    nodes = list(G.nodes)
    start, end = random.choice(nodes), random.choice(nodes)
    while start == end:
        end = random.choice(nodes)
    return start, end





def yen_ksp(G, source, target, K=1):
    A = [nx.shortest_path(G, source, target, weight='length')]
    B = []

    for k in range(1, K):
        for i in range(len(A[k - 1]) - 1):
            # Initialize the spur node and root path
            spur_node = A[k - 1][i]
            root_path = A[k - 1][:i + 1]

            # Keep track of removed edges to restore them later
            edges_removed = []

            # Remove the edges involved in the shortest paths found so far
            for path in A:
                if len(path) > i and root_path == path[:i + 1]:
                    u, v = path[i], path[i + 1]
                    edge_keys = list(G[u][v].keys())  # Make a separate list of edge keys
                    for key in edge_keys:
                        edge_data = G[u][v][key]
                        G.remove_edge(u, v, key)
                        edges_removed.append((u, v, key, edge_data))

            spur_path = None
            try:
                # Compute a spur path from the spur node to the target
                spur_path = nx.shortest_path(G, spur_node, target, weight='length')
            except nx.NetworkXNoPath:
                pass

            # Total path is the concatenation of root path and spur path
            if spur_path is not None:
                total_path = root_path[:-1] + spur_path
                B.append(total_path)

            # Restore the edges that were removed
            for u, v, key, data in edges_removed:
                G.add_edge(u, v, key=key, **data)

        if not B:
            break  # If no spur paths are found, stop searching

        # Sort the potential k-shortest paths by their lengths
        B.sort(key=lambda path: sum(G[path[j]][path[j + 1]][0]['length'] for j in range(len(path) - 1)))
        # Add the shortest path among B to the list of k-shortest paths
        A.append(B[0])
        # Remove the shortest path added to A from list B
        B.pop(0)

    return A


def plot_graph(G, path, title):
    fig, ax = ox.plot_graph_route(G, path, route_color='green', route_linewidth=6, node_size=0, bgcolor='k')
    plt.show()




def dijkstra_end_node(G, start):
    # Initialize priority queue, distances, and previous node records
    queue = [(0, start)]
    distances = {node: float('infinity') for node in G.nodes}
    previous_nodes = {node: None for node in G.nodes}
    distances[start] = 0

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        for neighbor in G.neighbors(current_node):
            edge_data = G.get_edge_data(current_node, neighbor, 0)
            edge_speed = G.get_edge_data(current_node, neighbor, 0).get('maxspeed', 30)
            edge_length = edge_data.get('length', 0)
            candidate_distance = current_distance + edge_length / (edge_speed / 3.6)

            if candidate_distance < distances[neighbor]:
                distances[neighbor] = candidate_distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (candidate_distance, neighbor))
    
    # Find the farthest node from the start node
    furthest_node = max(distances, key=distances.get)
    if distances[furthest_node] == float('infinity'):
        # return None if the farthest node cannot be reached
        return None, []

    # Reconstruct the path from start to the furthest node
    path = [furthest_node]
    while previous_nodes[furthest_node] is not None:
        furthest_node = previous_nodes[furthest_node]
        path.append(furthest_node)
    path.reverse()

    return path[-1], path

# The `double_sweep` function starts from here

def double_sweep(G, start=None):
    if start is None:
        start = random.choice(list(G.nodes))

    # First sweep to find the furthest node from the start
    _, path_to_furthest = dijkstra_end_node(G, start)
    # If no path found in first sweep, terminate
    if not path_to_furthest:
        return None

    # Second sweep from the furthest node found in the first sweep
    end_of_path, path = dijkstra_end_node(G, path_to_furthest[-1])

    return path

def plot_heatmap(G, algorithm_attr):
    # Get attribute values
    edge_attributes = list(nx.get_edge_attributes(G, algorithm_attr).values())
    
    # If attribute values are empty or not set, print a message and return
    if not edge_attributes:
        print(f"No data for attribute '{algorithm_attr}' found on edges.")
        return
    
    # Normalize attribute values
    norm = mcolors.Normalize(vmin=min(edge_attributes), vmax=max(edge_attributes))
    cmap = plt.get_cmap('hot')
    
    # Apply colormap normalization to edge attributes for coloring
    edge_colors = [cmap(norm(attr_value)) for attr_value in edge_attributes]
    
    # Plot graph
    fig, ax = ox.plot_graph(
        G, 
        node_size=0, 
        edge_color=edge_colors, 
        edge_linewidth=3, 
        bgcolor='k',
        show=False,  # show=False to further customize the plot before showing
    )
    
    # Add colorbar based on normalization
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label(algorithm_attr)
    plt.show()








def initialize_edge_usage(G):
    #Initialize or reset 'algorithm_uses' attribute for all edges to 0.
    nx.set_edge_attributes(G, 0, 'algorithm_uses')

def update_edge_usage(G, pred):
    # Reset 'algorithm_uses' to 0 for all edges
    initialize_edge_usage(G)
    
    # Iterate over all pairs of source and target nodes
    for source in G.nodes:
        for target in G.nodes:
            # Traverse the shortest path from target to source
            while target in pred[source] and pred[source][target] is not None:
                prev = pred[source][target]
                
                if prev is not None:
                    # Increment 'algorithm_uses' by accessing the edge data directly
                    # This works for both MultiGraphs and Graphs
                    if G.is_multigraph(): 
                        # For MultiGraphs, increment all edges between prev and target
                        for key in G[prev][target]:
                            if 'algorithm_uses' in G[prev][target][key]:
                                G[prev][target][key]['algorithm_uses'] += 1
                            else:
                                G[prev][target][key]['algorithm_uses'] = 1
                    else:
                        # For simple Graphs
                        if 'algorithm_uses' in G[prev][target]:
                            G[prev][target]['algorithm_uses'] += 1
                        else:
                            G[prev][target]['algorithm_uses'] = 1
                target = prev


def johnsons_algorithm_simplified(G):
    distances = {}
    predecessors = {}

    # Run Dijkstra's algorithm for each node
    for node in G.nodes():
        dist, pred = nx.single_source_dijkstra(G, source=node, weight='length')
        distances[node] = dist
        predecessors[node] = pred

    return distances, predecessors





    

def set_speed_weigths(G):
    for edge in G.edges:
        # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
        defaultSpeed = 30
        maxspeed = defaultSpeed
        miles_to_km = 1.60934
        G.edges[edge]['missingSpeedData'] = False
        if "maxspeed" in G.edges[edge]:
            maxspeed = G.edges[edge]["maxspeed"]

            if maxspeed == 'walk':
                maxspeed = 5
            if type(maxspeed) == list:
                for speed in maxspeed :
                    if type(speed) == str:
                        speeds = []
                        current_speed = speed
                        if speed == 'walk':
                            current_speed = int(current_speed.replace("walk", "5"))
                        if(type(current_speed) == str and 'mph' in current_speed):
                            current_speed = int(current_speed.replace("mph", "")) * miles_to_km
                        speeds.append(current_speed)
                if speeds:
                    maxspeed = min(speeds)
                else:
                    maxspeed = defaultSpeed
            elif type(maxspeed) == str and 'mph' in maxspeed:
                maxspeed = int(maxspeed.replace("mph", "")) * miles_to_km
        else:
            G.edges[edge]['missingSpeedData'] = True
            maxspeed = defaultSpeed
        G.edges[edge]["maxspeed"] = float(maxspeed)
        # Adding the "weight" attribute (time = distance / speed)
        G.edges[edge]["weight"] = G.edges[edge]["length"] / float(maxspeed)
    return G
def main():

    test_number = 'Test5'
    G = get_test_map(test_number)
    print("Pobrano mape testowa")

    G = set_speed_weigths(G)
    print("Ustawiono wagi grafu")

    # Create a client connection to your MongoDB server
    client = MongoClient('mongodb://localhost:27017/')

    # Connect to your database
    db = client['PracaMagisterska']

    collection = db[test_number]

    print(ox.basic_stats(G))
    
    start_nodes, end_nodes = get_start_end_nodes(test_number)

    if test_number in ['Test1', 'Test2', 'Test3', 'Test4', 'Test5']:
        result = tests_one_one(G, start_nodes, end_nodes)


    collection.insert_one(result)
    
    
    if test_number == 'Test6' or test_number == 'Test7':
        initialize_spfa_edge_usage(G)
        initialize_bellman_ford_edge_usage(G)
        for i in range(len(start_nodes)):
            start_node = start_nodes[i]
            end_node = end_nodes[i]

            start_time = time.time()
            distances, pred = bellman_ford(G, start_node)
            end_time = time.time()
            bellman_ford_algorithm_time = end_time - start_time

            distSum = 0
            bellman_ford_finite_length_paths_count = 0
            for end in end_nodes:
                if distances[end] != float('inf'):
                    distSum += distances[end]
                    count += 1
            if count > 0:
                bellman_ford_dist_average = distSum / bellman_ford_finite_length_paths_count
            else:
                bellman_ford_dist_average = float('inf')

            start_time = time.time()
            distances = spfa(G, start_node)
            end_time = time.time()
            spfa_algorithm_time = end_time - start_time

            distSum = 0
            spfa_finite_length_paths_count = 0
            for end in end_nodes:
                if distances[end] != float('inf'):
                    distSum += distances[end]
                    count += 1
            if count > 0:
                spfa_dist_average = distSum / spfa_finite_length_paths_count
            else:
                spfa_dist_average = float('inf')

            result = {}
            if distances:
                result.update({
                    f"Bellman Ford Time": bellman_ford_algorithm_time,
                    f"Bellman Ford Average Distance": bellman_ford_dist_average,
                    f"Bellman Ford Finite Length Paths Count": bellman_ford_finite_length_paths_count,
                    f"SPFA Time": spfa_algorithm_time,
                    f"SPFA Average Distance": spfa_dist_average,
                    f"SPFA Finite Length Paths Count": spfa_finite_length_paths_count
                    })
                collection.insert_one(result)
        plot_heatmap(G, 'spfa_algorithm_uses')
        spfa_sum = 0
        bellman_ford_sum = 0
        for u,v,data in G.edges(data=True):
            spfa_sum += data['spfa_algorithm_uses']
            bellman_ford_sum += data['bellman_ford_algorithm_uses']
        print('SPFA Iteracje: ', spfa_sum)
        plot_heatmap(G, 'bellman_ford_algorithm_uses')
        print('Bellman Ford Iteracje: ', bellman_ford_sum)

            

    if test_number == 'Test10' or test_number == 'Test11':
        # Ensure the Graph has a pos attribute for plotting
        G = ox.project_graph(G)

        # Initialize 'algorithm_uses' for all edges to 0
        initialize_edge_usage(G)
        # Run Floyd-Warshall algorithm
        start_time = time.time()
        dist, pred = floyd_warshall(G)
        end_time = time.time()
        floyd_warshall_algorithm_time = end_time - start_time
        
        # Update edge usage based on the Floyd-Warshall algorithm
        if dist and pred:
            update_edge_usage(G, pred)
            # Plot heatmap
            plot_heatmap(G, 'algorithm_uses')

        
        # To be placed within the `main` function, replacing the previous heatmap plotting section

        # Initialize 'length' for all edges to zero
        initialize_edge_usage(G)

        start_time = time.time()
        distances, predecessors = johnson(G)
        end_time = time.time()
        johnsons_algorithm_time = end_time - start_time

        if distances and predecessors:
            update_edge_usage(G, pred)
            # Plot heatmap
            plot_heatmap(G, 'algorithm_uses')

        print("Floyd-Warshall Time: ", floyd_warshall_algorithm_time)
        print("Johnson's Algorithm Time: ", johnsons_algorithm_time)

        floyd_warshall_sum = 0
        johnsons_algorithm_sum = 0
        for u,v,data in G.edges(data=True):
            floyd_warshall_sum += data['algorithm_uses']
            johnsons_algorithm_sum += data['algorithm_uses']
        print('Floyd-Warshall Iteracje: ', floyd_warshall_sum)
        print('Johnson Iteracje: ', johnsons_algorithm_sum)

        collection.insert_one({
            "Floyd-Warshall Time": floyd_warshall_algorithm_time,
            "Johnson's Algorithm Time": johnsons_algorithm_time,
            "Floyd-Warshall Iterations": floyd_warshall_sum,
            "Johnson's Algorithm Iterations": johnsons_algorithm_sum
        })

        plot_heatmap(G, 'algorithm_uses')

'''
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

       
    double_sweep_path = double_sweep(G, start)
    if double_sweep_path:
        travel_time, path_length, default_speed_distance, average_speed = analyze_path(G, double_sweep_path)
        print("Double Sweep Algorithm Results:")
        print(f"Travel Time: {travel_time} seconds")
        print(f"Path Length: {path_length} meters")
        print(f"Default Speed Distance: {default_speed_distance} meters")
        print(f"Average Speed: {average_speed} km/h")
        plot_graph(G, double_sweep_path, 'Double Sweep Route')

'''
############################

'''
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

       
    double_sweep_path = double_sweep(G, start)
    if double_sweep_path:
        travel_time, path_length, default_speed_distance, average_speed = analyze_path(G, double_sweep_path)
        print("Double Sweep Algorithm Results:")
        print(f"Travel Time: {travel_time} seconds")
        print(f"Path Length: {path_length} meters")
        print(f"Default Speed Distance: {default_speed_distance} meters")
        print(f"Average Speed: {average_speed} km/h")
        plot_graph(G, double_sweep_path, 'Double Sweep Route')

    
    
    


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

    
    # To be placed within the `main` function, replacing the previous heatmap plotting section

    # Initialize 'length' for all edges to zero
    initialize_edge_usage(G)

    distances, predecessors = johnsons_algorithm_simplified(G)

    if distances and predecessors:
        update_edge_usage(G, pred)
        # Plot heatmap
        plot_heatmap(G, 'algorithm_uses')
'''

if __name__ == '__main__':
    main()
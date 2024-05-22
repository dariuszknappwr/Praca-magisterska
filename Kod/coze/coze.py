from memory_profiler import profile
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
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
import matplotlib.colors as mcolors
import time
from pymongo import MongoClient



# Define a function to calculate time from distance and speed
def calculate_time(distance, max_speed):
    speed_m_per_s = max_speed / 3.6
    time = distance / speed_m_per_s
    return time

def download_map(city, osm_file_path):
    G = ox.graph_from_place(city, network_type='drive')
    ox.save_graphml(G, filepath=osm_file_path)
    return G

def load_local_map(osm_file_path):
    # Check if the file exists
    if os.path.isfile(osm_file_path):
        G = ox.load_graphml(osm_file_path)
        return G
    else:
        raise FileNotFoundError(f"OSM file cannot be found: {osm_file_path}")


def get_random_nodes(G):
    nodes = list(G.nodes)
    start, end = random.choice(nodes), random.choice(nodes)
    while start == end:
        end = random.choice(nodes)
    return start, end

#@profile
def dijkstra(G, orig, dest, style='length', plot=False, ):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    #for edge in G.edges:
        #style_unvisited_edge(G, edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(0, orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            #if plot:
                #print("Iterations:", step)
                #plot_graph()
            break
        if G.nodes[node]["visited"]: continue
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node):
            #style_visited_edge(G, (edge[0], edge[1], 0))
            neighbor = edge[1]
            if style == 'length':
                weight = G.edges[(edge[0], edge[1], 0)]["length"]
            else:
                weight = G.edges[(edge[0], edge[1], 0)]["weight"]

            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                #for edge2 in G.out_edges(neighbor):
                    #style_active_edge(G, (edge2[0], edge2[1], 0))
        step += 1

    # Path construction from end to start
    path = deque()
    current_node = dest
    while current_node is not None:
        path.appendleft(current_node)
        current_node = G.nodes[current_node]["previous"]
    return list(path), step

# A* finds a path from start to goal.
# h is the heuristic function. h(n) estimates the cost to reach goal from node n.
#@profile
def a_star(G, orig, dest, heuristic, style='length', plot=False):
    for node in G.nodes:
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
        G.nodes[node]["g_score"] = float("inf")
        G.nodes[node]["f_score"] = float("inf")
    if plot:
        for edge in G.edges:
            G = style_unvisited_edge(G, edge)
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    G.nodes[orig]["g_score"] = 0
    G.nodes[orig]["f_score"] = heuristic((G.nodes[orig]['x'], G.nodes[orig]['y']), (G.nodes[dest]['x'], G.nodes[dest]['y']))
    pq = [(G.nodes[orig]["f_score"], orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            if plot:
                print("Iterations:", step)
                plot_graph()
            break
        for edge in G.out_edges(node):
            if plot:
                G = style_visited_edge(edge[0], edge[1], 0)
            neighbor = edge[1]
            tentative_g_score = G.nodes[node]["g_score"] + heuristic((G.nodes[node]['x'], G.nodes[node]['y']), (G.nodes[neighbor]['x'], G.nodes[neighbor]['y']))
            if tentative_g_score < G.nodes[neighbor]["g_score"]:
                G.nodes[neighbor]["previous"] = node
                G.nodes[neighbor]["g_score"] = tentative_g_score
                G.nodes[neighbor]["f_score"] = tentative_g_score + heuristic((G.nodes[neighbor]['x'], G.nodes[neighbor]['y']), (G.nodes[dest]['x'], G.nodes[dest]['y']))
                heapq.heappush(pq, (G.nodes[neighbor]["f_score"], neighbor))
                if plot:
                    for edge2 in G.out_edges(neighbor):
                        G = style_active_edge((edge2[0], edge2[1], 0))
        step += 1

    path = deque()
    while dest is not None:
        path.appendleft(dest)
        dest = G.nodes[dest]['previous']

    #print("Number of iterations A*:", iteration_count)

    return list(path), step

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

def analyze_path(G, path):
    path_length = 0
    path_travel_time = 0
    missing_speed_data_distance = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        key = 0  # Assuming a simple graph
        edge_data = G.get_edge_data(u, v, key)
        
        if edge_data:
            edge_length = edge_data['length']
            edge_speed= edge_data['maxspeed']
            edge_travel_time = edge_length / (edge_speed / 3.6)  # travel time for edge
            path_length += edge_length
            path_travel_time += edge_travel_time

            # Check if a default speed was used
            if edge_data['missingSpeedData']:
                missing_speed_data_distance += edge_length

    average_speed = (path_length / path_travel_time * 3.6) if path_travel_time > 0 else 0  # Convert m/s to km/h

    return path_travel_time, path_length, missing_speed_data_distance, average_speed


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

def plot_graph(G):
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "#18080e"
    )

def style_unvisited_edge(G, edge):        
    G.edges[edge]["color"] = "blue"
    G.edges[edge]["alpha"] = 0.2
    G.edges[edge]["linewidth"] = 0.5
    return G

def style_visited_edge(G, edge):
    G.edges[edge]["color"] = "blue"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1
    return G

def style_active_edge(G, edge):
    G.edges[edge]["color"] = 'blue'
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1
    return G

def style_path_edge(G, edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1
    return G

def zero_heuristic(coord1, coord2):
    return 0

def infinity_heuristic(coord1, coord2):
    return float('inf')

def euclidean_heuristic(coord1, coord2):
    (x1, y1), (x2, y2) = coord1, coord2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def manhattan_heuristic(coord1, coord2):
    (x1, y1), (x2, y2) = coord1, coord2
    return abs(x1 - x2) + abs(y1 - y2)

def chebyshev_heuristic(coord1, coord2):
    (x1, y1), (x2, y2) = coord1, coord2
    return max(abs(x1 - x2), abs(y1 - y2))

def haversine(coord1, coord2):
    # Converts latitude and longitude to
    # spherical coordinates in radians.
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

def maxspeed_heuristic(coord1, coord2):
    (x1, y1), (x2, y2) = coord1, coord2
    return max(abs(x1 - x2), abs(y1 - y2))


def bellman_ford(G, source):
    # Step 1: Prepare distance and predecessor dictionaries
    distance = dict.fromkeys(G, float('infinity'))
    distance[source] = 0
    pred = {node: None for node in G}

    # Step 2: Relax edges repeatedly
    for _ in range(len(G) - 1):
        for u, v, data in G.edges(data=True):
            if distance[u] + data['length'] < distance[v]:
                distance[v] = distance[u] + data['length']
                pred[v] = u

    # Step 3: Check for negative weight cycles
    for u, v, data in G.edges(data=True):
        if distance[u] + data['length'] < distance[v]:
            raise nx.NetworkXUnbounded("Graph contains a negative weight cycle.")
    
    return distance, pred


def initialize_edge_usage(G):
    """
    Initialize or reset 'algorithm_uses' attribute for all edges to 0.
    """
    nx.set_edge_attributes(G, 0, 'algorithm_uses')






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



def spfa(G, start):
    initialize_edge_usage(G)
    
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start] = 0
    queue = deque([start])
    in_queue = {node: False for node in G.nodes()}
    in_queue[start] = True

    while queue:
        u = queue.popleft()
        in_queue[u] = False
        
        # Loop through each edge that comes out of `u`
        for v, adj_data in G.adj[u].items():
            for key, data in adj_data.items():  # Here we include the key in the iteration
                weight = data['length']
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    G.edges[u, v, key]['algorithm_uses'] += 1
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

    return distances


def floyd_warshall(G):
    # Initialize the distance and path matrices
    dist = {n: {m: float('inf') for m in G.nodes} for n in G.nodes}
    for n in G.nodes:
        dist[n][n] = 0
    pred = {n: {m: None for m in G.nodes} for n in G.nodes}

    # Initialize the distance to all edges that are present
    for u, v, data in G.edges(data=True):
        dist[u][v] = data['length']
        pred[u][v] = u
    
    for k in G.nodes:
        for i in G.nodes:
            for j in G.nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]

    # Check for negative weight cycles
    for n in G.nodes:
        if dist[n][n] < 0:
            print("Graph contains a negative-weight cycle")
            return None, None

    return dist, pred

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

    test_number = 'Test2'

    # Specify the path to your local .osm file
    if test_number == 'Test1':
        local_osm_file_path = 'Nowy_York_map.graphml'
    elif test_number == 'Test2':
        local_osm_file_path = 'Berlin_map.graphml'
    elif test_number == 'Test3':
        local_osm_file_path = 'Lubelskie_map.graphml'
    elif test_number == 'Test4':
        local_osm_file_path = 'Wroclaw_map.graphml'
    elif test_number == 'Test5':
        local_osm_file_path = 'Zalipie_map.graphml'
    try:
        G = load_local_map(local_osm_file_path)
    except FileNotFoundError as e:
        print(e)
        return

    G = set_speed_weigths(G)
    print("Ustawiono wagi grafu")

    # Create a client connection to your MongoDB server
    client = MongoClient('mongodb://localhost:27017/')

    # Connect to your database
    db = client['PracaMagisterska']

    collection = db[test_number]

    print(ox.basic_stats(G))
    
    start_nodes = []
    end_nodes = []
    with open(f'{test_number}_start_end_nodes.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            start, end = line.strip().split(',')
            start_nodes.append(int(start.strip()))
            end_nodes.append(int(end.strip()))

    if test_number == 'Test1' or 'Test2' or 'Test3' or 'Test4' or 'Test5':
        for i in range(len(start_nodes)):
            
            start_node = start_nodes[i]
            end_node = end_nodes[i]
            #start_node, end_node = get_random_nodes(G)
            # Perform the algorithm calculations using the start and end nodes

            algorithms = {
                "Dijkstra's": lambda G, start, end: dijkstra(G, start, end),
                "Dijkstra's Max Speed": lambda G, start, end: dijkstra(G, start, end, style='maxspeed'),
                "A Star Euclidean": lambda G, start, end: a_star(G, start, end, euclidean_heuristic),
                "A Star Manhattan": lambda G, start, end: a_star(G, start, end, manhattan_heuristic),
                "A Star Chebyshev": lambda G, start, end: a_star(G, start, end, chebyshev_heuristic),
                "A Star Haversine": lambda G, start, end: a_star(G, start, end, haversine),
            }

            result = {}
            for algorithm_name, algorithm_func in algorithms.items():
                start_time = time.time()
                path, iterations = algorithm_func(G, start_node, end_node)
                end_time = time.time()
                algorithm_time = end_time - start_time


                if path:
                    travel_time, path_length, default_speed_distance, average_speed = analyze_path(G, path)
                    result.update({
                        f"{algorithm_name} Time": algorithm_time,
                        f"{algorithm_name} Iterations": iterations,
                        f"{algorithm_name} Path": path,
                        f"{algorithm_name} Travel Time": travel_time,
                        f"{algorithm_name} Path Length": path_length,
                        f"{algorithm_name} Missing Speed Data Distance": default_speed_distance,
                        f"{algorithm_name} Average Speed": average_speed
                    })

            # Insert the result into the collection
            collection.insert_one(result)
    if test_number == 'Test6':
            distances = bellman_ford(G, start)
            if distances:
                plot_heatmap(G, 'algorithm_uses')

            distances = spfa(G, start)
            if distances:
                plot_heatmap(G, 'algorithm_uses')


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
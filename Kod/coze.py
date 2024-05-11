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


# Define a function to calculate time from distance and speed
def calculate_time(distance, max_speed):
    speed_m_per_s = max_speed / 3.6
    time = distance / speed_m_per_s
    time_hours = time / 3600
    return time_hours

def download_map(city):
    G = ox.graph_from_place(city, network_type='drive')
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

def get_edge_speed(G, u, v, key=0):
    default_speed = 30
    speed_data = G.edges[u, v, key].get('maxspeed', default_speed)

    if isinstance(speed_data, list):
        speed = int(speed_data[0].split()[0])
    elif isinstance(speed_data, str):
        if 'mph' in speed_data:
            speed = int(speed_data.split(' ')[0]) * 1.60934
        else:
            speed = int(speed_data.split()[0])
    elif isinstance(speed_data, (int, float)):
        speed = speed_data
    else:
        speed = default_speed

    return speed

def dijkstra(G, start, end):
    shortest_distances = {vertex: float('infinity') for vertex in G.nodes()}
    shortest_distances[start] = 0
    previous_vertices = {vertex: None for vertex in G.nodes()}
    unvisited = set(G.nodes())

    while unvisited:
        current = min(unvisited, key=lambda vertex: shortest_distances[vertex])
        unvisited.remove(current)

        if current == end or shortest_distances[current] == float('infinity'):
            break

        for neighbor in G.neighbors(current):
            for key, edge_data in G[current][neighbor].items():
                edge_length = edge_data.get('length', 0)
                edge_speed = get_edge_speed(G, current, neighbor, key)
                travel_time = edge_length / (edge_speed / 3.6)  # Convert speed to m/s
                
                if shortest_distances[current] + travel_time < shortest_distances[neighbor]:
                    shortest_distances[neighbor] = shortest_distances[current] + travel_time
                    previous_vertices[neighbor] = current

    # Reconstruct the shortest path
    path = []
    current_node = end
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_vertices[current_node]

    return path

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
    default_speed_distance = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        key = 0  # Assuming a simple graph
        edge_data = G.get_edge_data(u, v, key)
        
        if edge_data:
            edge_length = edge_data['length']
            edge_speed = get_edge_speed(G, u, v, key)
            edge_travel_time = edge_length / (edge_speed / 3.6)  # travel time for edge
            path_length += edge_length
            path_travel_time += edge_travel_time

            # Check if a default speed was used
            if 'maxspeed' not in edge_data or edge_data.get('maxspeed') == 'default_speed':
                default_speed_distance += edge_length

    average_speed = (path_length / path_travel_time * 3.6) if path_travel_time > 0 else 0  # Convert m/s to km/h

    return path_travel_time, path_length, default_speed_distance, average_speed


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
            edge_speed = get_edge_speed(G, current_node, neighbor, 0)
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


def a_star_algorithm(G, start, end, heuristic):
    # Initialize data structures
    queue = [(0, start)]  # Priority queue
    distances = {node: float('infinity') for node in G.nodes}  # Shortest distance from start to node
    previous_nodes = {node: None for node in G.nodes}  # Previous node in optimal path
    distances[start] = 0  # Distance from start to start is zero

    start_coord = (G.nodes[start]['y'], G.nodes[start]['x'])  # Start coordinates
    end_coord = (G.nodes[end]['y'], G.nodes[end]['x'])  # End coordinates
    costs = {start: heuristic(start_coord, end_coord)}  # Cost estimate for shortest path

    while queue:
        _, current_node = heapq.heappop(queue)
        if current_node == end:
            break

        current_coord = (G.nodes[current_node]['y'], G.nodes[current_node]['x'])
        for neighbor in G.adj[current_node]:
            edge_data = G.edges[current_node, neighbor, 0]
            edge_speed = get_edge_speed(G, current_node, neighbor, 0)
            edge_length = edge_data.get('length', 0)
            neighbor_coord = (G.nodes[neighbor]['y'], G.nodes[neighbor]['x'])
            candidate_distance = distances[current_node] + edge_length / (edge_speed / 3.6)  # Actual cost to neighbor
            estimated_heuristic = heuristic(neighbor_coord, end_coord)  # Heuristic cost from neighbor to end

            # Update cost if it's lower than previously found
            if candidate_distance + estimated_heuristic < costs.get(neighbor, float('infinity')):
                distances[neighbor] = candidate_distance
                costs[neighbor] = candidate_distance + estimated_heuristic
                heapq.heappush(queue, (costs[neighbor], neighbor))
                previous_nodes[neighbor] = current_node

    # Path reconstruction
    if previous_nodes[end] is None:
        return []  # No path found

    path = [end]
    while previous_nodes[path[-1]] is not None:
        path.append(previous_nodes[path[-1]])
    path.reverse()
    
    return path


def euclidean_heuristic(coord1, coord2):
    (x1, y1), (x2, y2) = coord1, coord2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def manhattan_heuristic(coord1, coord2):
    (x1, y1), (x2, y2) = coord1, coord2
    return abs(x1 - x2) + abs(y1 - y2)

def bellman_ford(G, start):
    # Initialize distances from the start to all nodes as infinity, and to start as 0
    distances = {vertex: float('infinity') for vertex in G.nodes}
    distances[start] = 0

    # Initialize edge usage count
    initialize_edge_usage(G)
    
    # Relax edges iteratively
    for _ in range(len(G) - 1):
        # Track if any edge got updated in this iteration
        updated = False
        for u, v, data in G.edges(data=True):
            weight = data['length']  # Assuming 'length' is the attribute for distance
            # Relax the edge
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                G.edges[u, v, 0]['algorithm_uses'] += 1  # Increment the usage for edge
                updated = True
        
        # If no update occured in this iteration, then no further updates will occur
        if not updated:
            break

    # Check for negative weight cycles
    for u, v, data in G.edges(data=True):
        weight = data['length']
        if distances[u] + weight < distances[v]:
            print("Graph contains a negative-weight cycle")
            return None

    return distances

def initialize_edge_usage(G):
    for u, v, key in G.edges(keys=True):
        G[u][v][key]['algorithm_uses'] = 0




def plot_heatmap(G, algorithm_attr):
    # Use OSMnx to visualize the edge usage
    edge_colors = ox.plot.get_edge_colors_by_attr(G, algorithm_attr, cmap="hot")
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=3,
        bgcolor='k'
    )
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

def initialize_edge_usage_heatmap(G, dist):
    for u, v, data in G.edges(data=True):
        if dist[u][v] != float('infinity'):
            if 'algorithm_uses' not in data:
                data['algorithm_uses'] = 0
            data['algorithm_uses'] += 1  # Increment the usage for edge

def plot_graph_with_heatmap(G):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Normalize edge usage for color mapping
    max_usage = max([data.get('algorithm_uses', 1) for u, v, data in G.edges(data=True)])
    edge_colors = [data.get('algorithm_uses', 0) / max_usage if max_usage > 0 else 0 for u, v, data in G.edges(data=True)]

    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        edge_cmap=plt.cm.hot,
        width=2,
        ax=ax
    )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(vmin=-max_usage, vmax=max_usage))
    sm._A = []
    
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Edge Usage Intensity')

    plt.title("Edge Usage Heatmap")
    plt.axis('off')
    plt.show()


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




def main():
    # Specify the path to your local .osm file
    local_osm_file_path = 'Chodel_map.graphml'
    try:
        G = load_local_map(local_osm_file_path)
    except FileNotFoundError as e:
        print(e)
        return
    
    start, end = get_random_nodes(G)
    
    '''
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

       
    double_sweep_path = double_sweep(G, start)
    # Inside your main function after the double_sweep call
    print(double_sweep_path)
    if double_sweep_path:
        travel_time, path_length, default_speed_distance, average_speed = analyze_path(G, double_sweep_path)
        print("Double Sweep Algorithm Results:")
        print(f"Travel Time: {travel_time} seconds")
        print(f"Path Length: {path_length} meters")
        print(f"Default Speed Distance: {default_speed_distance} meters")
        print(f"Average Speed: {average_speed} km/h")
        plot_graph(G, double_sweep_path, 'Double Sweep Route')

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
   '''



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

if __name__ == '__main__':
    main()
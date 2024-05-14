from graph_utils import get_edge_speed

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



from coze import get_edge_speed

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
                travel_time = edge_length / (edge_speed / 3.6)  # zamiana szybkości do m/s
                
                if shortest_distances[current] + travel_time < shortest_distances[neighbor]:
                    shortest_distances[neighbor] = shortest_distances[current] + travel_time
                    previous_vertices[neighbor] = current

    # Zrekonstruowanie najkrótszej ścieżki
    path = []
    current_node = end
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_vertices[current_node]

    return path

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
import heapq
import math
from graph_utils import get_edge_speed

def a_star_algorithm(G, start, end, heuristic):
    queue = [(0, start)]  # kolejka priorytetowa (koszt, wierzchołek)
    distances = {node: float('infinity') for node in G.nodes}  # najkrótsza odległość od startu
    previous_nodes = {node: None for node in G.nodes}  # poprzedni wierzchołek na najkrótszej ścieżce
    distances[start] = 0  # odległość od startu do startu wynosi 0

    start_coord = (G.nodes[start]['y'], G.nodes[start]['x'])
    end_coord = (G.nodes[end]['y'], G.nodes[end]['x'])
    costs = {start: heuristic(start_coord, end_coord)}  # koszt = odległość + heurystyka

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
            candidate_distance = distances[current_node] + edge_length / (edge_speed / 3.6)  # koszt podróży do sąsiada
            estimated_heuristic = heuristic(neighbor_coord, end_coord)  # koszt od sąsiada do celu przy wykorzystaniu heurystyki

            # Aktualizacjia kosztu jeżeli ma mniejszą wartość niż poprzedni znaleziony
            if candidate_distance + estimated_heuristic < costs.get(neighbor, float('infinity')):
                distances[neighbor] = candidate_distance
                costs[neighbor] = candidate_distance + estimated_heuristic
                heapq.heappush(queue, (costs[neighbor], neighbor))
                previous_nodes[neighbor] = current_node

    # Rekonstrukcja ścieżki
    if previous_nodes[end] is None:
        return []  # nie znaleziono ścieżki

    path = [end]
    while previous_nodes[path[-1]] is not None:
        path.append(previous_nodes[path[-1]])
    path.reverse()
    
    return path

def haversine(coord1, coord2):
    # Converts latitude and longitude to spherical coordinates in radians.
    # zamienia szerokość i długość geograficzną na współrzędne sferyczne w radianach
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)

    # Wzór Haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Promień Ziemi w Kilometrach
    return c * r


def euclidean_heuristic(coord1, coord2):
    (x1, y1), (x2, y2) = coord1, coord2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def manhattan_heuristic(coord1, coord2):
    (x1, y1), (x2, y2) = coord1, coord2
    return abs(x1 - x2) + abs(y1 - y2)

def chebyshev_heuristic(coord1, coord2):
    (x1, y1), (x2, y2) = coord1, coord2
    return max(abs(x1 - x2), abs(y1 - y2))

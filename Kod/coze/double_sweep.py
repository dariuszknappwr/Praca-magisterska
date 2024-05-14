import random
import heapq
from graph_utils import get_edge_speed

def double_sweep_ksp(G, start, end, K):
    paths = []

    # Kopia zapasowa oryginalnego grafu do przywrócenia
    original_edges = list(G.edges(data=True))
    
    for _ in range(K):
        path = double_sweep(G, start, end)
        if path:
            paths.append(path)
            # Usuwanie krawędzi z poprzedniej ścieżki
            G.remove_edges_from(zip(path[:-1], path[1:]))
        else:
            break  

    # Przywracanie oryginalnych krawędzi
    G.add_edges_from(original_edges)
    return paths

def double_sweep(G, start, end):
    if start is None:
        start = random.choice(list(G.nodes))

    # Pierwsze przeszukanie, aby znaleźć najdalszy węzeł od startu
    _, path_to_furthest = dijkstra_end_node(G, start, end)
    # Jeśli nie znaleziono ścieżki w pierwszym przeszukaniu, zakończ
    if not path_to_furthest:
        return None

    # Drugie przeszukanie od najdalszego węzła znalezionego w pierwszym przeszukaniu
    end_of_path, path = dijkstra_end_node(G, path_to_furthest[-1], path_to_furthest[0])

    return path

def dijkstra_end_node(G, start, end):
    queue = [(0, start)]
    distances = {node: float('infinity') for node in G.nodes}
    previous_nodes = {node: None for node in G.nodes}
    distances[start] = 0

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_node == end:
            break
        for neighbor in G.neighbors(current_node):
            edge_data = G.get_edge_data(current_node, neighbor, 0)
            edge_speed = get_edge_speed(G, current_node, neighbor, 0)
            edge_length = edge_data.get('length', 0)
            candidate_distance = current_distance + edge_length / (edge_speed / 3.6)

            if candidate_distance < distances[neighbor]:
                distances[neighbor] = candidate_distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (candidate_distance, neighbor))
    
    # Odtworzenie ścieżki od startu do węzła końcowego
    path = []
    while end is not None:
        path.append(end)
        end = previous_nodes[end]
    path.reverse()

    return path[-1], path





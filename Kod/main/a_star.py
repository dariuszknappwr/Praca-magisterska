from collections import deque
import heapq
from plot_graph import plot_graph, style_unvisited_edge, style_visited_edge, style_active_edge
import math
from profiler import profile
import tracemalloc
# A* znajduje ścieżkę od punktu startowego do celu.
# h to funkcja heurystyczna. h(n) szacuje koszt dotarcia do celu z węzła n.
#@profile
def a_star(G, orig, dest, heuristic, style='length', plot=False):
    tracemalloc.start()
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
                print("Iteracje:", step)
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

    #print("Liczba iteracji A*:", iteration_count)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return list(path), step, peak, 0

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
    # Konwertuje szerokość i długość geograficzną na
    # współrzędne sferyczne w radianach.
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)

    # Wzór Haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Promień Ziemi w kilometrach
    return c * r

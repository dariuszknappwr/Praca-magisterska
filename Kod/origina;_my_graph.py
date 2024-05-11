import osmnx as ox
import random
import heapq
import networkx as nx
import numpy as np



G = nx.MultiDiGraph()
# Add nodes with arbitrary coordinates
coordinates = {
    1: (0, 0),
    2: (1, 3),
    3: (2.5,4),
    4: (2.5,1),
    5: (2.5,-3),
    6: (5,3),
}

for i in range(1, 7):
    x, y = coordinates[i]
    G.add_node(i, x=x, y=y)

# Add edges with a 'length' attribute
G.add_edges_from([
    (1, 2, {'length': 3, 'maxspeed': 40, 'weight': 3/40}),
    (2, 3, {'length': 3, 'maxspeed': 40, 'weight': 3/40}),
    (1, 4, {'length': 4, 'maxspeed': 40, 'weight': 4/40}),
    (4, 3, {'length': 3, 'maxspeed': 40, 'weight': 3/40}),
    (1, 5, {'length': 5, 'maxspeed': 40, 'weight': 5/40}),
    (4, 5, {'length': 3, 'maxspeed': 40, 'weight': 3/40}),
    (4, 6, {'length': 5, 'maxspeed': 40, 'weight': 5/40}),
    (3, 6, {'length': 5, 'maxspeed': 40, 'weight': 5/40}),
    (5, 6, {'length': 2, 'maxspeed': 40, 'weight': 2/40}),
    (2, 5, {'length': 8, 'maxspeed': 40, 'weight': 8/40}),
    (5, 5, {'length': 1, 'maxspeed': 40, 'weight': 1/40})
])
G.graph['crs'] = "EPSG:4326"

for edge in G.edges:
    # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
    maxspeed = 40
    if "maxspeed" in G.edges[edge]:
        maxspeed = G.edges[edge]["maxspeed"]
        if type(maxspeed) == list:
            speeds = [ int(speed) for speed in maxspeed ]
            maxspeed = min(speeds)
        elif type(maxspeed) == str:
            maxspeed = int(maxspeed)
    G.edges[edge]["maxspeed"] = maxspeed
    # Adding the "weight" attribute (time = distance / speed)
    G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed



def style_unvisited_edge(edge):        
    G.edges[edge]["color"] = "blue"
    G.edges[edge]["alpha"] = 0.2
    G.edges[edge]["linewidth"] = 0.5

def style_visited_edge(edge):
    G.edges[edge]["color"] = "blue"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    G.edges[edge]["color"] = 'blue'
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def plot_graph():
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "#18080e"
    )

def plot_heatmap(algorithm):
    edge_colors = ox.plot.get_edge_colors_by_attr(G, f"{algorithm}_uses", cmap="hot")
    fig, _ = ox.plot_graph(
        G,
        node_size = 0,
        edge_color = edge_colors,
        bgcolor = "#18080e"
    )



def dijkstra(orig, dest, plot=False):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(0, orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            if plot:
                print("Iteraciones:", step)
                plot_graph()
            return
        if G.nodes[node]["visited"]: continue
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node):
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1

def distance(node1, node2):
    x1, y1 = G.nodes[node1]["x"], G.nodes[node1]["y"]
    x2, y2 = G.nodes[node2]["x"], G.nodes[node2]["y"]
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def a_star(orig, dest, plot=False):
    for node in G.nodes:
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
        G.nodes[node]["g_score"] = float("inf")
        G.nodes[node]["f_score"] = float("inf")
    for edge in G.edges:
        style_unvisited_edge(edge)
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    G.nodes[orig]["g_score"] = 0
    G.nodes[orig]["f_score"] = distance(orig, dest)
    pq = [(G.nodes[orig]["f_score"], orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            if plot:
                print("Iteraciones:", step)
                plot_graph()
            return
        for edge in G.out_edges(node):
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]
            tentative_g_score = G.nodes[node]["g_score"] + distance(node, neighbor)
            if tentative_g_score < G.nodes[neighbor]["g_score"]:
                G.nodes[neighbor]["previous"] = node
                G.nodes[neighbor]["g_score"] = tentative_g_score
                G.nodes[neighbor]["f_score"] = tentative_g_score + distance(neighbor, dest)
                heapq.heappush(pq, (G.nodes[neighbor]["f_score"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1

def yen_algorithm(orig, dest, K):
    A = [dijkstra(orig, dest)]
    B = []
    for k in range(1, K):
        for i in range(len(A[k-1])-1):
            spur_node = A[k-1][i]
            root_path = A[k-1][:i+1]
            for path in A:
                if root_path == path[:i+1]:
                    G.edges[path[i:i+2]]["weight"] = float("inf")
            spur_path = dijkstra(spur_node, dest)
            if spur_path:
                total_path = root_path[:-1] + spur_path
                B.append(total_path)
            for path in A:
                G.edges[path[i:i+2]]["weight"] = G.edges[path[i:i+2]]["length"] / G.edges[path[i:i+2]]["maxspeed"]
        if not B:
            break
        B.sort(key=lambda path: sum(G.edges[path[i:i+2]]["weight"] for i in range(len(path)-1)))
        A.append(B[0])
        B = []
    return A

def reconstruct_path(orig, dest, plot=False, algorithm=None):
    for edge in G.edges:
        style_unvisited_edge(edge)
    dist = 0
    speeds = []
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        dist += G.edges[(prev, curr, 0)]["length"]
        speeds.append(G.edges[(prev, curr, 0)]["maxspeed"])
        style_path_edge((prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000
    if plot:
        print(f"Distance: {dist}")
        print(f"Avg. speed: {sum(speeds)/len(speeds)}")
        print(f"Total time: {dist/(sum(speeds)/len(speeds)) * 60}")
        plot_graph()



start = random.choice(list(G.nodes))
end = random.choice(list(G.nodes))

dijkstra(1, 6, plot=True)

reconstruct_path(1, 6, plot=True)

a_star(start, end, plot=True)

reconstruct_path(start, end, plot=True)


N = 100 # times to run each algorithm
for edge in G.edges:
    G.edges[edge]["dijkstra_uses"] = 0
    G.edges[edge]["a_star_uses"] = 0

for _ in range(N): # (might take a while, depending on N)
    start = random.choice(list(G.nodes))
    end = random.choice(list(G.nodes))
    dijkstra(start, end)
    reconstruct_path(start, end, algorithm="dijkstra")
    a_star(start, end)
    reconstruct_path(start, end, algorithm="a_star")

plot_heatmap("dijkstra")

plot_heatmap("a_star")

K = 3
yen_paths = yen_algorithm(1, 6, K)
for i, path in enumerate(yen_paths):
    print(f"Path {i+1}: {path}")
    reconstruct_path(path[0], path[-1], plot=True)


from collections import deque
import heapq

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
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



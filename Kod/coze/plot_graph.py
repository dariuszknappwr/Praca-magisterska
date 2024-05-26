import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


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
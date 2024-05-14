import matplotlib.pyplot as plt
import osmnx as ox
import matplotlib.colors as mcolors
import networkx as nx

def plot_graph(G, path, title):
    fig, ax = ox.plot_graph_route(G, path, route_color='green', route_linewidth=6, node_size=0, bgcolor='k')
    plt.show()

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
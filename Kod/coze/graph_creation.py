import osmnx as ox
import networkx as nx
import os

def download_map(city):
    G = ox.graph_from_place(city, network_type='drive')
    return G

def load_local_map(osm_file_path):
    # Check if the file exists
    if os.path.isfile(osm_file_path):
        G = ox.load_graphml(osm_file_path)
        return G
    else:
        raise FileNotFoundError(f"OSM file cannot be found: {osm_file_path}")

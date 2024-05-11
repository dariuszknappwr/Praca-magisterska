import osmnx as ox
import networkx as nx
from itertools import islice
from matplotlib import pyplot as plt

# Create a street network graph for a city
G = ox.graph_from_place('Chodel, Lubelskie, Polska', network_type='drive')

#save graph to file
ox.save_graphml(G, filepath='Chodel_map.graphml')
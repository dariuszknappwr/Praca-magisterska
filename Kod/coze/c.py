import osmnx as ox
import networkx as nx


#to jeszcze nie załadowane
G = ox.graph_from_place("zachodniopomorskie, Poland", network_type='drive')
print("Załadowano graf z OSM")
existing_graph = ox.load_graphml("merged_map.graphml")
merged_graph = nx.compose(existing_graph, G)
ox.save_graphml(merged_graph, filepath="merged_map.graphml")
print("Zapisano graf do pliku")
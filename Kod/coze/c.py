import osmnx as ox
import networkx as nx


#to jeszcze nie załadowane
#G = ox.graph_from_place("zachodniopomorskie, Poland", network_type='drive')
#print("Załadowano graf z OSM")
#existing_graph = ox.load_graphml("merged_map.graphml")
#merged_graph = nx.compose(existing_graph, G)
#ox.save_graphml(merged_graph, filepath="merged_map.graphml")
#print("Zapisano graf do pliku")

G = ox.graph_from_place("Płońsk, Mazowieckie, Poland", network_type='drive')
ox.save_graphml(G, filepath="Plonsk_map.graphml")
print("Zapisano graf do pliku")
#print out number of vertices and edges
print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())

import osmnx as ox

# Definiowanie miejsca do pobrania danych
place_name = "Wrocław, Dolnośląskie, Poland"

# Pobieranie grafu drogowego dla zdefiniowanego miejsca
G = ox.graph_from_place(place_name, network_type='drive')

# Wyświetlanie podstawowych informacji o grafie
print(ox.basic_stats(G))
# 'n':4501, 'm':9699, 'k_avg':4.31, 'edge_length_total':1145418.03 ...

# Rysowanie grafu
ox.plot_graph(G, node_size=0, edge_linewidth=0.5)

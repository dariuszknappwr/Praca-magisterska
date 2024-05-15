import osmnx as ox
import networkx as nx
from itertools import islice
from matplotlib import pyplot as plt
import random

def plot_graph(G, path, title):
    fig, ax = ox.plot_graph_route(G, path, route_color='green', route_linewidth=6, node_size=0, bgcolor='k')
    plt.show()

# Create a street network graph for a city
#G = ox.graph_from_place('New York', network_type='drive')

#save graph to file
#ox.save_graphml(G, filepath='Nowy_York_map.graphml')
'''
def get_random_nodes(G):
    nodes = list(G.nodes)
    start, end = random.choice(nodes), random.choice(nodes)
    while start == end:
        end = random.choice(nodes)
    return start, end
# Load graph from file
G = ox.load_graphml(filepath='Nowy_York_map.graphml')

# print out number of vertices and edges
print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())
    
start, end = get_random_nodes(G)
# Generate 100 random start and end nodes
start_end_pairs = []
for _ in range(100):
    start_node, end_node = get_random_nodes(G)
    start_end_pairs.append((start_node, end_node))

# Save start and end nodes to file
with open('start_end_nodes_test1.txt', 'w') as file:
    for start_node, end_node in start_end_pairs:
        file.write(f"{start_node},{end_node}\n")

'''
#load data from mongo and calculate statistics

import pymongo
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')

# Connect to your database
db = client['PracaMagisterska']

########## Test 1 ##########
collection = db['Test1']

results = collection.find()
results = list(collection.find())

times_euclidean = []
times_manhattan = []
times_chebyshev = []
times_haversine = []
times_dijkstra = []

for result in results:
    times_euclidean.append(result['A* Algorithm Time Euclidean'])
    times_manhattan.append(result['A* Algorithm Time Manhattan'])
    times_chebyshev.append(result['A* Algorithm Time Chebyshev'])
    times_haversine.append(result['A* Algorithm Time Haversine'])
    times_dijkstra.append(result['Dijkstra\'s Algorithm Time'])
    
# Calculate statistics
print("Test 1 Statistics:")
print("A* Algorithm Euclidean Heuristic Average Time:", sum(times_euclidean) / len(times_euclidean))
print("A* Algorithm Manhattan Heuristic Average Time:", sum(times_manhattan) / len(times_manhattan))
print("A* Algorithm Chebyshev Heuristic Average Time:", sum(times_chebyshev) / len(times_chebyshev))
print("A* Algorithm Haversine Heuristic Average Time:", sum(times_haversine) / len(times_haversine))
print("Dijkstra's Algorithm Average Time:", sum(times_dijkstra) / len(times_dijkstra))

# now calculates medians and other statistics
import statistics
print("A* Algorithm Euclidean Heuristic Median Time:", statistics.median(times_euclidean))
print("A* Algorithm Manhattan Heuristic Median Time:", statistics.median(times_manhattan))
print("A* Algorithm Chebyshev Heuristic Median Time:", statistics.median(times_chebyshev))
print("A* Algorithm Haversine Heuristic Median Time:", statistics.median(times_haversine))
print("Dijkstra's Algorithm Median Time:", statistics.median(times_dijkstra))

print("A* Algorithm Euclidean Heuristic Standard Deviation Time:", statistics.stdev(times_euclidean))
print("A* Algorithm Manhattan Heuristic Standard Deviation Time:", statistics.stdev(times_manhattan))
print("A* Algorithm Chebyshev Heuristic Standard Deviation Time:", statistics.stdev(times_chebyshev))
print("A* Algorithm Haversine Heuristic Standard Deviation Time:", statistics.stdev(times_haversine))
print("Dijkstra's Algorithm Standard Deviation Time:", statistics.stdev(times_dijkstra))

print("A* Algorithm Euclidean Heuristic Variance Time:", statistics.variance(times_euclidean))
print("A* Algorithm Manhattan Heuristic Variance Time:", statistics.variance(times_manhattan))
print("A* Algorithm Chebyshev Heuristic Variance Time:", statistics.variance(times_chebyshev))
print("A* Algorithm Haversine Heuristic Variance Time:", statistics.variance(times_haversine))
print("Dijkstra's Algorithm Variance Time:", statistics.variance(times_dijkstra))

travel_times_euclidean = []
travel_times_manhattan = []
travel_times_chebyshev = []
travel_times_haversine = []
travel_times_dijkstra = []

for result in results:
    travel_times_euclidean.append(result['A* Algorithm Euclidean Travel Time'])
    travel_times_manhattan.append(result['A* Algorithm Manhattan Travel Time'])
    travel_times_chebyshev.append(result['A* Algorithm Chebyshev Travel Time'])
    travel_times_haversine.append(result['A* Algorithm Haversine Travel Time'])
    travel_times_dijkstra.append(result['Dijkstra\'s Algorithm Travel Time'])

# Calculate statistics
print("Test 1 Travel Times Statistics:")
print("A* Algorithm Euclidean Heuristic Average Travel Time:", sum(travel_times_euclidean) / len(travel_times_euclidean))
print("A* Algorithm Manhattan Heuristic Average Travel Time:", sum(travel_times_manhattan) / len(travel_times_manhattan))
print("A* Algorithm Chebyshev Heuristic Average Travel Time:", sum(travel_times_chebyshev) / len(travel_times_chebyshev))
print("A* Algorithm Haversine Heuristic Average Travel Time:", sum(travel_times_haversine) / len(travel_times_haversine))
print("Dijkstra's Algorithm Average Travel Time:", sum(travel_times_dijkstra) / len(travel_times_dijkstra))

# now calculates medians and other statistics
print("A* Algorithm Euclidean Heuristic Median Travel Time:", statistics.median(travel_times_euclidean))
print("A* Algorithm Manhattan Heuristic Median Travel Time:", statistics.median(travel_times_manhattan))
print("A* Algorithm Chebyshev Heuristic Median Travel Time:", statistics.median(travel_times_chebyshev))
print("A* Algorithm Haversine Heuristic Median Travel Time:", statistics.median(travel_times_haversine))
print("Dijkstra's Algorithm Median Travel Time:", statistics.median(travel_times_dijkstra))

print("A* Algorithm Euclidean Heuristic Standard Deviation Travel Time:", statistics.stdev(travel_times_euclidean))
print("A* Algorithm Manhattan Heuristic Standard Deviation Travel Time:", statistics.stdev(travel_times_manhattan))
print("A* Algorithm Chebyshev Heuristic Standard Deviation Travel Time:", statistics.stdev(travel_times_chebyshev))
print("A* Algorithm Haversine Heuristic Standard Deviation Travel Time:", statistics.stdev(travel_times_haversine))
print("Dijkstra's Algorithm Standard Deviation Travel Time:", statistics.stdev(travel_times_dijkstra))

print("A* Algorithm Euclidean Heuristic Variance Travel Time:", statistics.variance(travel_times_euclidean))
print("A* Algorithm Manhattan Heuristic Variance Travel Time:", statistics.variance(travel_times_manhattan))
print("A* Algorithm Chebyshev Heuristic Variance Travel Time:", statistics.variance(travel_times_chebyshev))
print("A* Algorithm Haversine Heuristic Variance Travel Time:", statistics.variance(travel_times_haversine))
print("Dijkstra's Algorithm Variance Travel Time:", statistics.variance(travel_times_dijkstra))

'''
Test 1 Statistics:
A* Algorithm Euclidean Heuristic Average Time: 0.41292064428329467
A* Algorithm Manhattan Heuristic Average Time: 0.37360220432281493
A* Algorithm Chebyshev Heuristic Average Time: 0.38998674392700194
A* Algorithm Haversine Heuristic Average Time: 0.512901418209076
Dijkstra's Algorithm Average Time: 161.9589048743248
A* Algorithm Euclidean Heuristic Median Time: 0.38529086112976074
A* Algorithm Manhattan Heuristic Median Time: 0.34880805015563965
A* Algorithm Chebyshev Heuristic Median Time: 0.3679927587509155
A* Algorithm Haversine Heuristic Median Time: 0.4872092008590698
Dijkstra's Algorithm Median Time: 177.2908718585968
A* Algorithm Euclidean Heuristic Standard Deviation Time: 0.2493303579330087
A* Algorithm Manhattan Heuristic Standard Deviation Time: 0.2239814985082165
A* Algorithm Chebyshev Heuristic Standard Deviation Time: 0.2341985081398613
A* Algorithm Haversine Heuristic Standard Deviation Time: 0.3152825290975243
Dijkstra's Algorithm Standard Deviation Time: 83.71878055022583
A* Algorithm Euclidean Heuristic Variance Time: 0.06216562738700223
A* Algorithm Manhattan Heuristic Variance Time: 0.05016771167398619
A* Algorithm Chebyshev Heuristic Variance Time: 0.05484894121493667
A* Algorithm Haversine Heuristic Variance Time: 0.09940307315413127
Dijkstra's Algorithm Variance Time: 7008.83421681687
Test 1 Travel Times Statistics:
A* Algorithm Euclidean Heuristic Average Travel Time: 1606.221425583444
A* Algorithm Manhattan Heuristic Average Travel Time: 1606.221425583444
A* Algorithm Chebyshev Heuristic Average Travel Time: 1606.221425583444
A* Algorithm Haversine Heuristic Average Travel Time: 1606.221425583444
Dijkstra's Algorithm Average Travel Time: 1606.4103263750728
A* Algorithm Euclidean Heuristic Median Travel Time: 1494.4508709362667
A* Algorithm Manhattan Heuristic Median Travel Time: 1494.4508709362667
A* Algorithm Chebyshev Heuristic Median Travel Time: 1494.4508709362667
A* Algorithm Haversine Heuristic Median Travel Time: 1494.4508709362667
Dijkstra's Algorithm Median Travel Time: 1494.4508709362667
A* Algorithm Euclidean Heuristic Standard Deviation Travel Time: 839.2181573395197
A* Algorithm Manhattan Heuristic Standard Deviation Travel Time: 839.2181573395197
A* Algorithm Chebyshev Heuristic Standard Deviation Travel Time: 839.2181573395197
A* Algorithm Haversine Heuristic Standard Deviation Travel Time: 839.2181573395197
Dijkstra's Algorithm Standard Deviation Travel Time: 839.2617028206148
A* Algorithm Euclidean Heuristic Variance Travel Time: 704287.115608339
A* Algorithm Manhattan Heuristic Variance Travel Time: 704287.115608339
A* Algorithm Chebyshev Heuristic Variance Travel Time: 704287.115608339
A* Algorithm Haversine Heuristic Variance Travel Time: 704287.115608339
Dijkstra's Algorithm Variance Travel Time: 704360.2058213579
'''
'''
# load file Nowy_York_map.graphml and check all edges, check if there are any edges with negative weights and print them
G = ox.load_graphml(filepath='Nowy_York_map.graphml')
for u, v, data in G.edges(data=True):
    if data['length'] < 0:
        print(f"Edge ({u}, {v}) has negative weight: {data['length']}")
'''
'''
path_lengths_euclidean = []
path_lengths_manhattan = []
path_lengths_chebyshev = []
path_lengths_haversine = []
path_lengths_dijkstra = []

for result in results:
    path_lengths_euclidean.append(result['A* Algorithm Euclidean Path Length'])
    path_lengths_manhattan.append(result['A* Algorithm Manhattan Path Length'])
    path_lengths_chebyshev.append(result['A* Algorithm Chebyshev Path Length'])
    path_lengths_haversine.append(result['A* Algorithm Haversine Path Length'])
    path_lengths_dijkstra.append(result['Dijkstra\'s Algorithm Path Length'])

# Calculate statistics
print("Test 1 Path Length Statistics:")
print("A* Algorithm Euclidean Heuristic Average Path Length:", sum(path_lengths_euclidean) / len(path_lengths_euclidean))
print("A* Algorithm Manhattan Heuristic Average Path Length:", sum(path_lengths_manhattan) / len(path_lengths_manhattan))
print("A* Algorithm Chebyshev Heuristic Average Path Length:", sum(path_lengths_chebyshev) / len(path_lengths_chebyshev))
print("A* Algorithm Haversine Heuristic Average Path Length:", sum(path_lengths_haversine) / len(path_lengths_haversine))
print("Dijkstra's Algorithm Average Path Length:", sum(path_lengths_dijkstra) / len(path_lengths_dijkstra))

# now calculates medians and other statistics
print("A* Algorithm Euclidean Heuristic Median Path Length:", statistics.median(path_lengths_euclidean))
print("A* Algorithm Manhattan Heuristic Median Path Length:", statistics.median(path_lengths_manhattan))
print("A* Algorithm Chebyshev Heuristic Median Path Length:", statistics.median(path_lengths_chebyshev))
print("A* Algorithm Haversine Heuristic Median Path Length:", statistics.median(path_lengths_haversine))
print("Dijkstra's Algorithm Median Path Length:", statistics.median(path_lengths_dijkstra))

print("A* Algorithm Euclidean Heuristic Standard Deviation Path Length:", statistics.stdev(path_lengths_euclidean))
print("A* Algorithm Manhattan Heuristic Standard Deviation Path Length:", statistics.stdev(path_lengths_manhattan))
print("A* Algorithm Chebyshev Heuristic Standard Deviation Path Length:", statistics.stdev(path_lengths_chebyshev))
print("A* Algorithm Haversine Heuristic Standard Deviation Path Length:", statistics.stdev(path_lengths_haversine))
print("Dijkstra's Algorithm Standard Deviation Path Length:", statistics.stdev(path_lengths_dijkstra))

print("A* Algorithm Euclidean Heuristic Variance Path Length:", statistics.variance(path_lengths_euclidean))
print("A* Algorithm Manhattan Heuristic Variance Path Length:", statistics.variance(path_lengths_manhattan))
print("A* Algorithm Chebyshev Heuristic Variance Path Length:", statistics.variance(path_lengths_chebyshev))
print("A* Algorithm Haversine Heuristic Variance Path Length:", statistics.variance(path_lengths_haversine))
print("Dijkstra's Algorithm Variance Path Length:", statistics.variance(path_lengths_dijkstra))
'''

average_speeds_euclidean = []
average_speeds_manhattan = []
average_speeds_chebyshev = []
average_speeds_haversine = []
average_speeds_dijkstra = []

for result in results:
    average_speeds_euclidean.append(result['A* Algorithm Euclidean Average Speed'])
    average_speeds_manhattan.append(result['A* Algorithm Manhattan Average Speed'])
    average_speeds_chebyshev.append(result['A* Algorithm Chebyshev Average Speed'])
    average_speeds_haversine.append(result['A* Algorithm v Average Speed'])
    average_speeds_dijkstra.append(result['Dijkstra\'s Algorithm Average Speed'])

# Calculate statistics
print("Test 1 Average Speed Statistics:")
print("A* Algorithm Euclidean Heuristic Average Average Speed:", sum(average_speeds_euclidean) / len(average_speeds_euclidean))
print("A* Algorithm Manhattan Heuristic Average Average Speed:", sum(average_speeds_manhattan) / len(average_speeds_manhattan))
print("A* Algorithm Chebyshev Heuristic Average Average Speed:", sum(average_speeds_chebyshev) / len(average_speeds_chebyshev))
print("A* Algorithm Haversine Heuristic Average Average Speed:", sum(average_speeds_haversine) / len(average_speeds_haversine))
print("Dijkstra's Algorithm Average Average Speed:", sum(average_speeds_dijkstra) / len(average_speeds_dijkstra))

# now calculates medians and other statistics
print("A* Algorithm Euclidean Heuristic Median Average Speed:", statistics.median(average_speeds_euclidean))
print("A* Algorithm Manhattan Heuristic Median Average Speed:", statistics.median(average_speeds_manhattan))
print("A* Algorithm Chebyshev Heuristic Median Average Speed:", statistics.median(average_speeds_chebyshev))
print("A* Algorithm Haversine Heuristic Median Average Speed:", statistics.median(average_speeds_haversine))
print("Dijkstra's Algorithm Median Average Speed:", statistics.median(average_speeds_dijkstra))

print("A* Algorithm Euclidean Heuristic Standard Deviation Average Speed:", statistics.stdev(average_speeds_euclidean))
print("A* Algorithm Manhattan Heuristic Standard Deviation Average Speed:", statistics.stdev(average_speeds_manhattan))
print("A* Algorithm Chebyshev Heuristic Standard Deviation Average Speed:", statistics.stdev(average_speeds_chebyshev))
print("A* Algorithm Haversine Heuristic Standard Deviation Average Speed:", statistics.stdev(average_speeds_haversine))
print("Dijkstra's Algorithm Standard Deviation Average Speed:", statistics.stdev(average_speeds_dijkstra))

print("A* Algorithm Euclidean Heuristic Variance Average Speed:", statistics.variance(average_speeds_euclidean))
print("A* Algorithm Manhattan Heuristic Variance Average Speed:", statistics.variance(average_speeds_manhattan))
print("A* Algorithm Chebyshev Heuristic Variance Average Speed:", statistics.variance(average_speeds_chebyshev))
print("A* Algorithm Haversine Heuristic Variance Average Speed:", statistics.variance(average_speeds_haversine))
print("Dijkstra's Algorithm Variance Average Speed:", statistics.variance(average_speeds_dijkstra))

default_speeds_dijkstra = []
path_lengths_dijkstra = []

for result in results:
    path_lengths_dijkstra.append(result['Dijkstra\'s Algorithm Path Length'])
    default_speeds_dijkstra.append(result['Dijkstra\'s Algorithm Default Speed Distance'])

# plot graphs
plt.plot(default_speeds_dijkstra, label='Suma odcinków z domyślną prędkością')
plt.plot(path_lengths_dijkstra, label='Badana trasa')
plt.title('Porównanie długości trasy i sumy odcinków o domyślnej prędkości w zależności od numeru testu dla algorytmu Dijkstry')
plt.xlabel('Numer testu')
plt.ylabel('Długość trasy [metry]')
plt.legend()
plt.show()





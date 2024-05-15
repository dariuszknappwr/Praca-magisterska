import osmnx as ox
import networkx as nx
from itertools import islice
from matplotlib import pyplot as plt
import random
from pymongo import MongoClient
import statistics

client = MongoClient('mongodb://localhost:27017/')

# Connect to your database
db = client['PracaMagisterska']

########## Test 1 ##########
collection = db['Test1']

results = list(collection.find())

algorithms = ['A* Algorithm Euclidean', 'A* Algorithm Manhattan', 'A* Algorithm Chebyshev', 'A* Algorithm Haversine', 'Dijkstra\'s Algorithm']
statistics_names = ["Time", "Travel Time", "Path Length", "Default Speed Distance", "Average Speed"]

# Initialize a nested dictionary to store the times for each algorithm and each statistic
times = {algorithm: {stat: [] for stat in statistics_names} for algorithm in algorithms}

for result in results:
    for algorithm in algorithms:
        for stat in statistics_names:
            times[algorithm][stat].append(result[algorithm + ' ' + stat])

# Now you can calculate and print the mean time for each algorithm and each statistic
for algorithm in algorithms:
    for stat in statistics_names:
        time_list = times[algorithm][stat]
        if time_list:  # Check if the list is not empty
            print(f"{algorithm} {stat} Average:", sum(time_list) / len(time_list))
        else:
            print(f"{algorithm} {stat} Average: No data")
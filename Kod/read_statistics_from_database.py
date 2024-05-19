import osmnx as ox
import networkx as nx
from itertools import islice
from matplotlib import pyplot as plt
import random
from pymongo import MongoClient
import statistics
import csv

client = MongoClient('mongodb://localhost:27017/')

# Connect to your database
db = client['PracaMagisterska']

########## Test 1 ##########
collection = db['Test2']

results = list(collection.find())

algorithms = ['Dijkstra\'s', 'Dijkstra\'s Max Speed', 'A Star Euclidean', 'A Star Manhattan', 'A Star Chebyshev', 'A Star Haversine']
statistics_names = ["Time", "Iterations", "Travel Time", "Path Length", "Default Speed Distance", "Average Speed"]

# Initialize a nested dictionary to store the times for each algorithm and each statistic
times = {algorithm: {stat: [] for stat in statistics_names} for algorithm in algorithms}
for result in results:
    for algorithm in algorithms:
        for stat in statistics_names:
            times[algorithm][stat].append(result[algorithm + ' ' + stat])

# Plot a figure for each statistic
for stat in statistics_names:
    fig, ax = plt.subplots()

    # Create a list to store the values for the statistic
    stat_values = []

    # Calculate the value for each algorithm
    for algorithm in algorithms:
        value_list = times[algorithm][stat]
        if value_list:  # Check if the list is not empty
            value = round(sum(value_list) / len(value_list), 2)
            stat_values.append(value)
        else:
            stat_values.append(0)

    # Plot the values
    ax.bar(algorithms, stat_values)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(stat)
    ax.set_title(f'{stat} for each algorithm in Test 2')

    # Save the plot to a file
    plt.savefig(f'Test2_{stat}_plot.png')

    # Show the plot
    plt.show()


# Open the CSV file in write mode
with open('Test2_statistics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Now you can calculate and write the statistics to the CSV file
    for algorithm in algorithms:
        for stat in statistics_names:
            time_list = times[algorithm][stat]
            if time_list:  # Check if the list is not empty
                average = round(sum(time_list) / len(time_list), 2)
                median = round(statistics.median(time_list), 2)
                std_dev = round(statistics.stdev(time_list), 2)
                variance = round(statistics.variance(time_list), 2)
                writer.writerow([algorithm, stat, average, median, std_dev, variance])
            else:
                writer.writerow([algorithm, stat, 'No data', '', '', ''])
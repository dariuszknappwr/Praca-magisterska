from matplotlib import pyplot as plt
from pymongo import MongoClient
import statistics
from collections import defaultdict
import csv


client = MongoClient('mongodb://localhost:27017/')

# Connect to your database
db = client['PracaMagisterska']

for test_number in ['Test21']:
    collection = db[test_number]

    results = list(collection.find())

    algorithms = ['Yen\'s', 'Hoffman-Pavley', 'Dijkstra']

    statistics_names = []

    # Open the CSV file in write mode
    with open(f'{test_number}/statistics.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(['Algorithm', 'K', 'Time', 'Memory', 'CPU'])

        # Iterate over the results
        for result in results:
            for algorithm in algorithms:
                for k in [1,2,3,4,5,10,25,50,100]:
                    # Extract the statistics for this algorithm and K
                    time = result.get(f"{algorithm} time - K={k}")
                    memory = result.get(f"{algorithm} memory - K={k}")
                    cpu = result.get(f"{algorithm} CPU - K={k}")

                    # Write the statistics to the CSV file
                    writer.writerow([algorithm, k, time, memory, cpu])
    # Initialize dictionaries to store the total and count of each statistic
    totals = defaultdict(lambda: defaultdict(dict))
    counts = defaultdict(lambda: defaultdict(dict))

    # Iterate over the results
    for result in results:
        for algorithm in algorithms:
            for k in [1,2,3,4,5,10,25,50,100]:
                # Extract the statistics for this algorithm and K
                time = result.get(f"{algorithm} time - K={k}")
                memory = result.get(f"{algorithm} memory - K={k}")
                cpu = result.get(f"{algorithm} CPU - K={k}")
                if result.get(f"{algorithm} Algorithm - K={k}") != None:
                    max_found_k = len(result.get(f"{algorithm} Algorithm - K={k}"))
                    travel_time = result.get(f"{algorithm} Algorithm - K={k}")[max_found_k-1].get('Travel Time')
                    path_length = result.get(f"{algorithm} Algorithm - K={k}")[max_found_k-1].get('Path Length')
                    default_speed_distance = result.get(f"{algorithm} Algorithm - K={k}")[max_found_k-1].get('Default Speed Distance')
                    average_speed = result.get(f"{algorithm} Algorithm - K={k}")[max_found_k-1].get('Average Speed')

                    # Add the statistics to the totals and increment the counts
                    for statistic, value in [('Time', time), ('Memory', memory), ('CPU', cpu), 
                                            ('Travel Time', travel_time), ('Path Length', path_length), 
                                            ('Default Speed Distance', default_speed_distance), 
                                            ('Average Speed', average_speed)]:
                        if value is not None:
                            totals[algorithm][k][statistic] = totals[algorithm][k].get(statistic, 0) + value
                            counts[algorithm][k][statistic] = counts[algorithm][k].get(statistic, 0) + 1

        # Open the CSV file in write mode
        with open(f'{test_number}/average_statistics.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(['Algorithm', 'K', 'Time', 'Travel Time', 'Path Length', 'Default Speed Distance','Average Speed'])

            # Iterate over the totals and counts
            for algorithm, k_values in totals.items():
                for k, statistics in k_values.items():
                    stats_tmp = []
                    for statistic, total in statistics.items():
                        count = counts[algorithm][k][statistic]
                        average = total / count
                        stats_tmp.append(round(average,2))
                        # Write the average statistic to the CSV file
                    writer.writerow([algorithm, k, stats_tmp[0], stats_tmp[1], stats_tmp[2], stats_tmp[3], stats_tmp[4], average])

        # write all results to csv file
        with open(f'{test_number}/all_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(['Algorithm', 'K', 'Time', 'Memory', 'CPU', 'Travel Time', 'Path Length', 'Default Speed Distance','Average Speed'])

            # Iterate over the results
            for result in results:
                for algorithm in algorithms:
                    for k in [1,2,3,4,5,10,25,50,100]:
                        # Extract the statistics for this algorithm and K
                        time = result.get(f"{algorithm} time - K={k}")
                        memory = result.get(f"{algorithm} memory - K={k}")
                        cpu = result.get(f"{algorithm} CPU - K={k}")
                        if result.get(f"{algorithm} Algorithm - K={k}") != None:
                            max_found_k = len(result.get(f"{algorithm} Algorithm - K={k}"))
                            travel_time = result.get(f"{algorithm} Algorithm - K={k}")[max_found_k-1].get('Travel Time')
                            path_length = result.get(f"{algorithm} Algorithm - K={k}")[max_found_k-1].get('Path Length')
                            default_speed_distance = result.get(f"{algorithm} Algorithm - K={k}")[max_found_k-1].get('Default Speed Distance')
                            average_speed = result.get(f"{algorithm} Algorithm - K={k}")[max_found_k-1].get('Average Speed')
                            # Write the statistics to the CSV file
                            writer.writerow([algorithm, k, time, memory, cpu, travel_time, path_length, default_speed_distance, average_speed])
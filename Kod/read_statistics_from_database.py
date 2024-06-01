from matplotlib import pyplot as plt
from pymongo import MongoClient
import statistics
from collections import defaultdict
import csv


client = MongoClient('mongodb://localhost:27017/')

# Connect to your database
db = client['PracaMagisterska']
'''
for test_number in ['Test1', 'Test2', 'Test3', 'Test4', 'Test5']:
    collection = db[test_number]

    results = list(collection.find())

    algorithms = ['Dijkstra\'s', 'Dijkstra\'s Max Speed', 'Dijkstra\'s Fibonacci', 'A Star Euclidean', 'A Star Manhattan', 'A Star Chebyshev', 'A Star Haversine']
    statistics_names = ["Time", "Iterations", "Travel Time", "Path Length", "Missing Speed Data Distance", "Average Speed"]

    # Initialize a nested dictionary to store the times for each algorithm and each statistic
    i = 0
    times = {algorithm: {stat: [] for stat in statistics_names} for algorithm in algorithms}
    for result in results:
        i += 1
        print(i)
        for algorithm in algorithms:
            for stat in statistics_names:
                times[algorithm][stat].append(result[algorithm + ' ' + stat])
        
                

    # Plot a figure for each statistic
    for stat in statistics_names:
        fig, ax = plt.subplots()

        # Create a list to store the values for the statistic
        average_values = []
        median_values = []
        std_dev_values = []
        variance_values = []

        # Calculate the value for each algorithm
        for algorithm in algorithms:
            value_list = times[algorithm][stat]
            if value_list:  # Check if the list is not empty
                average_values.append(round(sum(value_list) / len(value_list), 2))
                median_values.append(round(statistics.median(value_list),2))
                std_dev_values.append(round(statistics.stdev(value_list),2))
                variance_values.append(round(statistics.variance(value_list),2))
            else:
                average_values.append(0)
                median_values.append(0)
                std_dev_values.append(0)
                variance_values.append(0)

        # Plot the values
        
        ax.bar(algorithms, average_values, color='blue')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(stat)
        ax.set_title(f'{stat} for each algorithm in {test_number}')

        # Save the plot to a file
        plt.savefig(f'{test_number}/{test_number}_{stat}_plot.png')
        plt.close()

        # Show the plot
        #plt.show()

    optimal_algorithm_count = {
        "Dijkstra's": 0,
        "Dijkstra's Max Speed": 0,
        "Dijkstra's Fibonacci": 0,
        "A Star Euclidean": 0,
        "A Star Manhattan": 0,
        "A Star Chebyshev": 0,
        "A Star Haversine": 0
    }

    satisfactory_algorithm_count = {
        "Dijkstra's": 0,
        "Dijkstra's Max Speed": 0,
        "Dijkstra's Fibonacci": 0,
        "A Star Euclidean": 0,
        "A Star Manhattan": 0,
        "A Star Chebyshev": 0,
        "A Star Haversine": 0
    }

    path_efficiency = {
        "Dijkstra's": 0,
        "Dijkstra's Max Speed": 0,
        "Dijkstra's Fibonacci": 0,
        "A Star Euclidean": 0,
        "A Star Manhattan": 0,
        "A Star Chebyshev": 0,
        "A Star Haversine": 0
    }

    speed_efficiency = {
        "Dijkstra's": 0,
        "Dijkstra's Max Speed": 0,
        "Dijkstra's Fibonacci": 0,
        "A Star Euclidean": 0,
        "A Star Manhattan": 0,
        "A Star Chebyshev": 0,
        "A Star Haversine": 0
    }

    for result in results:
        optimal_path_length = result["Dijkstra's Path Length"]
        optimal_speed = result["Dijkstra's Max Speed Average Speed"]
        for algorithm in algorithms:
            path_length = result[algorithm + ' Path Length']
            path_length_percentage = path_length / optimal_path_length
            if path_length_percentage <= 1:
                optimal_algorithm_count[algorithm] += 1
            if path_length_percentage <= 1.10:
                satisfactory_algorithm_count[algorithm] += 1
            path_efficiency[algorithm] += optimal_path_length / path_length
            speed_efficiency[algorithm] += result[algorithm + ' Average Speed'] / optimal_speed
            



    # Open the CSV file in write mode
    with open(f'{test_number}/{test_number}_statistics.csv', 'w', newline='', encoding='utf-8') as csvfile:
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
            writer.writerow([algorithm, 'Percent of optimal path length', round(optimal_algorithm_count[algorithm]/len(results),2), '', '', ''])
            writer.writerow([algorithm, 'Percent of satisfactory path length', round(satisfactory_algorithm_count[algorithm]/len(results),2), '', '', ''])
            writer.writerow([algorithm, 'Path efficiency', round(path_efficiency[algorithm]/len(results),2), '', '', ''])
            writer.writerow([algorithm, 'Speed efficiency', round(speed_efficiency[algorithm]/len(results),2), '', '', ''])

        

    # Open the CSV file in write mode
    with open(f'{test_number}/{test_number}_statistics_table.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        header = ['Mierzona wartość'] + algorithms
        writer.writerow(header)

        # Now you can calculate and write the statistics to the CSV file
        for stat in statistics_names:
            row = [stat]
            for algorithm in algorithms:
                time_list = times[algorithm][stat]
                if time_list:  # Check if the list is not empty
                    average = round(sum(time_list) / len(time_list), 2)
                    std_dev = round(statistics.stdev(time_list), 2)
                    row.append(f'{average} ± {std_dev}')
                else:
                    row.append('No data')
            writer.writerow(row)

        # Open the CSV file in write mode
    with open(f'{test_number}/{test_number}_statistics_table_mean.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        header = ['Mierzona wartość'] + algorithms
        writer.writerow(header)

        # Now you can calculate and write the statistics to the CSV file
        for stat in statistics_names:
            row = [stat]
            for algorithm in algorithms:
                time_list = times[algorithm][stat]
                if time_list:  # Check if the list is not empty
                    average = round(sum(time_list) / len(time_list), 2)
                    row.append(f'{average}')
                else:
                    row.append('No data')
            writer.writerow(row)

    with open(f'{test_number}/{test_number}_statistics_all_values.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        header = ['Mierzona wartość'] + ['Czas'] + ['Ilość iteracji'] + ['Czas podróży'] + ['Długość ścieżki'] + ['Brakujące dane o prędkości'] + ['Średnia prędkość']
        writer.writerow(header)

        # Now you can calculate and write the statistics to the CSV file
        for stat in statistics_names:
            row = [stat]
            for algorithm in ['Dijkstra\'s Max Speed']:
                for x in times[algorithm][stat]:
                    row.append(x)
                else:
                    row.append('No data')
            writer.writerow(row)
'''








'''
for test_number in ['Test6']:
    collection = db[test_number]

    results = list(collection.find())

    algorithms = ['Bellman Ford', 'SPFA']

    statistics_names = ["Time", "Distance Sum", "Average Distance", "Finite Length Paths Count", "Consumed Memory", "Consumed CPU"]

    # Initialize a nested dictionary to store the times for each algorithm and each statistic
    i = 0
    times = {algorithm: {stat: [] for stat in statistics_names} for algorithm in algorithms}
    for result in results:
        i += 1
        print(i)
        for algorithm in algorithms:
            for stat in statistics_names:
                times[algorithm][stat].append(result[algorithm + ' ' + stat])



    # Open the CSV file in write mode

    with open(f'{test_number}/{test_number}_statistics.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Now you can calculate and write the statistics to the CSV file
        for algorithm in algorithms:
            for stat in statistics_names:
                i = 0
                time_list = times[algorithm][stat]
                if time_list:
                    if time_list[i] == None:
                        continue
                    average = round(sum(time_list) / len(time_list), 2)
                    median = round(statistics.median(time_list), 2)
                    std_dev = round(statistics.stdev(time_list), 2)
                    variance = round(statistics.variance(time_list), 2)
                    writer.writerow([algorithm, stat, average, median, std_dev, variance])
                else:
                    writer.writerow([algorithm, stat, 'No data', '', '', ''])
                i += 1

    # Open the CSV file in write mode
    with open(f'{test_number}/{test_number}_statistics_table.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        header = ['Mierzona wartość'] + algorithms
        writer.writerow(header)

        # Now you can calculate and write the statistics to the CSV file
        for stat in statistics_names:
            row = [stat]
            i = 0
            for algorithm in algorithms:
                time_list = times[algorithm][stat]
                if time_list:
                    if time_list[i] == None:
                        continue
                    average = round(sum(time_list) / len(time_list), 2)
                    std_dev = round(statistics.stdev(time_list), 2)
                    row.append(f'{average} ± {std_dev}')
                else:
                    row.append('No data')
                i += 1
            writer.writerow(row)

        # Open the CSV file in write mode
    with open(f'{test_number}/{test_number}_statistics_table_mean.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        header = ['Mierzona wartość'] + algorithms
        writer.writerow(header)

        # Now you can calculate and write the statistics to the CSV file
        for stat in statistics_names:
            row = [stat]
            i = 0
            for algorithm in algorithms:
                time_list = times[algorithm][stat]
                if time_list:
                    if time_list[i] == None:
                        continue
                    average = round(sum(time_list) / len(time_list), 2)
                    row.append(f'{average}')
                else:
                    row.append('No data')
                i += 1
            writer.writerow(row)

    with open(f'{test_number}/{test_number}_statistics_all_values.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        header = ['Mierzona wartość'] + ['Czas'] + ['Suma odległości'] + ['Średnia odległość'] + ['Ilość ścieżek skończonych'] + ['Zużyta pamięć'] + ['Zużyty CPU']
        writer.writerow(header)

        # Now you can calculate and write the statistics to the CSV file
        for stat in statistics_names:
            row = [stat]
            for algorithm in ['Bellman Ford']:
                for x in times[algorithm][stat]:
                    row.append(x)
                else:
                    row.append('No data')
            writer.writerow(row)
'''
for test_number in ['Test18']:
    collection = db[test_number]

    results = list(collection.find())

    algorithms = ['Yen\'s', 'Hoffman-Pavley', 'Dijkstra']

    statistics_names = []

    # Open the CSV file in write mode
    with open('statistics.csv', 'w', newline='') as file:
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
    with open('average_statistics.csv', 'w', newline='') as file:
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

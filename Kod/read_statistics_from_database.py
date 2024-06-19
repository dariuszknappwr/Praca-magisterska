from matplotlib import pyplot as plt
from pymongo import MongoClient
import statistics
from collections import defaultdict
import csv


client = MongoClient('mongodb://localhost:27017/')

# Connect to your database
db = client['PracaMagisterska']

for test_number in ['Test1', 'Test2', 'Test3', 'Test4', 'Test5']:
    collection = db[f"{test_number}_memory"]

    results = list(collection.find())

    algorithms = ['Dijkstra\'s', 'Dijkstra\'s Max Speed', 'Dijkstra\'s Fibonacci', 'A Star Euclidean', 'A Star Manhattan', 'A Star Chebyshev', 'A Star Haversine']
    statistics_names = ["Time", "Consumed Memory", "Iterations", "Travel Time", "Path Length", "Missing Speed Data Distance", "Average Speed"]

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










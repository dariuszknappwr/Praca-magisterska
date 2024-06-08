from matplotlib import pyplot as plt
from pymongo import MongoClient
import statistics
from collections import defaultdict
import csv


client = MongoClient('mongodb://localhost:27017/')

# Connect to your database
db = client['PracaMagisterska']
for test_number in ['Test11', 'Test12', 'Test13', 'Test14', 'Test15', 'Test16', 'Test17']:
    collection = db[test_number]

    results = list(collection.find())

    algorithms = ['Floyd-Warshall', 'Johnson\'s']

    statistics_names = ["Time", "Iterations"]#, "Consumed Memory", "Consumed CPU"]

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

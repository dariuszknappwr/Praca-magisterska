from dijkstra import dijkstra
from a_star import a_star, euclidean_heuristic, manhattan_heuristic, chebyshev_heuristic, haversine
from graph_utils import analyze_path
import time


def tests_one_one(G, start, end):
    algorithms = {
    "Dijkstra's": lambda G, start, end: dijkstra(G, start, end),
    "Dijkstra's Max Speed": lambda G, start, end: dijkstra(G, start, end, style='maxspeed'),
    "A Star Euclidean": lambda G, start, end: a_star(G, start, end, euclidean_heuristic),
    "A Star Manhattan": lambda G, start, end: a_star(G, start, end, manhattan_heuristic),
    "A Star Chebyshev": lambda G, start, end: a_star(G, start, end, chebyshev_heuristic),
    "A Star Haversine": lambda G, start, end: a_star(G, start, end, haversine),
}

    result = {}
    for algorithm_name, algorithm_func in algorithms.items():
        start_time = time.time()
        
        function_data, consumed_memory, consumed_cpu = algorithm_func(G, start, end)
        path, iterations = function_data
        
        end_time = time.time()
        
        algorithm_time = end_time - start_time

        if path:
            travel_time, path_length, default_speed_distance, average_speed = analyze_path(G, path)
            result.update({
            f"{algorithm_name} Time": algorithm_time,
            f"{algorithm_name} Iterations": iterations,
            f"{algorithm_name} Path": path,
            f"{algorithm_name} Travel Time": travel_time,
            f"{algorithm_name} Path Length": path_length,
            f"{algorithm_name} Missing Speed Data Distance": default_speed_distance,
            f"{algorithm_name} Average Speed": average_speed,
            f"{algorithm_name} Consumed Memory": consumed_memory,
            f"{algorithm_name} Consumed CPU": consumed_cpu
        })

    return result
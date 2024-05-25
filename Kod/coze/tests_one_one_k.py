from yen import yen_ksp
from double_sweep import double_sweep
from graph_utils import analyze_path
from plot_graph import plot_graph

def tests_one_one_k(G, start, end, K, plot=False):
    results = {}
    ksp_paths = yen_ksp(G, start, end, K)
    for i, path in enumerate(ksp_paths, start=1):
        ksp_results = analyze_path(G, path)
        
        print(f"Yen's KSP Algorithm - Path {i} Results:")
        print(f"Travel Time: {ksp_results[0]} seconds")
        print(f"Path Length: {ksp_results[1]} meters")
        print(f"Default Speed Distance: {ksp_results[2]} meters")
        print(f"Average Speed: {ksp_results[3]} km/h")
        if plot:
            plot_graph(G, path, f"Yen's KSP Route {i}")

    
    double_sweep_path = double_sweep(G, start)
    if double_sweep_path:
        travel_time, path_length, default_speed_distance, average_speed = analyze_path(G, double_sweep_path)
        print("Double Sweep Algorithm Results:")
        print(f"Travel Time: {travel_time} seconds")
        print(f"Path Length: {path_length} meters")
        print(f"Default Speed Distance: {default_speed_distance} meters")
        print(f"Average Speed: {average_speed} km/h")
        if plot:
            plot_graph(G, double_sweep_path, 'Double Sweep Route')
    
    return 
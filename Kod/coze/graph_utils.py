import networkx as nx

def analyze_path(G, path):
    path_length = 0
    path_travel_time = 0
    missing_speed_data_distance = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        key = 0  # Assuming a simple graph
        edge_data = G.get_edge_data(u, v, key)
        
        if edge_data:
            edge_length = edge_data['length']
            edge_speed= edge_data['maxspeed']
            edge_travel_time = edge_length / (edge_speed / 3.6)  # travel time for edge
            path_length += edge_length
            path_travel_time += edge_travel_time

            # Check if a default speed was used
            if edge_data['missingSpeedData']:
                missing_speed_data_distance += edge_length

    average_speed = (path_length / path_travel_time * 3.6) if path_travel_time > 0 else 0  # Convert m/s to km/h

    return path_travel_time, path_length, missing_speed_data_distance, average_speed



def set_speed_weigths(G):
    for edge in G.edges:
        # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
        defaultSpeed = 30
        maxspeed = defaultSpeed
        miles_to_km = 1.60934
        G.edges[edge]['missingSpeedData'] = False
        if "maxspeed" in G.edges[edge]:
            maxspeed = G.edges[edge]["maxspeed"]

            if maxspeed == 'walk':
                maxspeed = 5
            if type(maxspeed) == list:
                for speed in maxspeed :
                    if type(speed) == str:
                        speeds = []
                        current_speed = speed
                        if speed == 'walk':
                            current_speed = int(current_speed.replace("walk", "5"))
                        if(type(current_speed) == str and 'mph' in current_speed):
                            current_speed = int(current_speed.replace("mph", "")) * miles_to_km
                        speeds.append(current_speed)
                if speeds:
                    maxspeed = min(speeds)
                else:
                    maxspeed = defaultSpeed
            elif type(maxspeed) == str and 'mph' in maxspeed:
                maxspeed = int(maxspeed.replace("mph", "")) * miles_to_km
        else:
            G.edges[edge]['missingSpeedData'] = True
            maxspeed = defaultSpeed
        G.edges[edge]["maxspeed"] = float(maxspeed)
        # Adding the "weight" attribute (time = distance / speed)
        G.edges[edge]["weight"] = G.edges[edge]["length"] / float(maxspeed)
    return G

def initialize_edge_usage(G):
    #Initialize or reset 'algorithm_uses' attribute for all edges to 0.
    nx.set_edge_attributes(G, 0, 'algorithm_uses')

def update_edge_usage(G, pred, G_all_uses=None):
    # Reset 'algorithm_uses' to 0 for all edges
    initialize_edge_usage(G)
    
    # Iterate over all pairs of source and target nodes
    for source in G.nodes:
        for target in G.nodes:
            # Traverse the shortest path from target to source
            while target in pred[source] and pred[source][target] is not None:
                prev = pred[source][target]
                
                if prev is not None:
                    # Increment 'algorithm_uses' by accessing the edge data directly
                    # This works for both MultiGraphs and Graphs
                    if G.is_multigraph(): 
                        if isinstance(prev, list):
                            prev = tuple(prev)
                        if isinstance(target, list):
                            target = tuple(target)
                        # For MultiGraphs, increment all edges between prev and target
                        if prev in G and target in G[prev]:
                            for key in G[prev][target]:
                                if 'algorithm_uses' in G[prev][target][key]:
                                    G[prev][target][key]['algorithm_uses'] += 1
                                    #G_all_uses[prev][target][key]['algorithm_uses'] += 1
                                else:
                                    print(f"Error: 'algorithm_uses' not set for edge ({prev}, {target})")
                    else:
                        # For simple Graphs
                        if 'algorithm_uses' in G[prev][target]:
                            G[prev][target]['algorithm_uses'] += 1
                            #G_all_uses[prev][target]['algorithm_uses'] += 1
                        else:
                            print(f"Error: 'algorithm_uses' not set for edge ({prev}, {target})")
                target = prev
def update_edge_usage_johnson(G, sources_targets_map):
    #print(f"Dist map: {dist_map}")
    # Reset 'algorithm_uses' to 0 for all edges
    initialize_edge_usage(G)
    
    # Iterate over all pairs of source and target nodes
    for sources in sources_targets_map:
        # funkcja zwraca również wierzchołki None - None, które należy pominąć
        if sources == None:
            continue
        for source, paths in sources.items():
            prev_path_node = None
            for target_path in paths.items():
                target = target_path[0]
                path = target_path[1]
                for v in path:
                    if v == source:
                        prev_path_node = path[0]
                        continue
                    if 'algorithm_uses' in G[prev_path_node][v][0]:
                        G[prev_path_node][v][0]['algorithm_uses'] = G[prev_path_node][v][0]['algorithm_uses'] + 1
                    else:
                        print(f"Error: 'algorithm_uses' not set for edge ({prev_path_node}, {v})")
                    prev_path_node = v
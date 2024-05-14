import random
import networkx as nx

# Define a function to calculate time from distance and speed
def calculate_time(distance, max_speed):
    speed_m_per_s = max_speed / 3.6
    time = distance / speed_m_per_s
    time_hours = time / 3600
    return time_hours

def get_random_nodes(G):
    nodes = list(G.nodes)
    start, end = random.choice(nodes), random.choice(nodes)
    while start == end:
        end = random.choice(nodes)
    return start, end

def get_edge_speed(G, u, v, key=0):
    default_speed = 30
    speed_data = G.edges[u, v, key].get('maxspeed', default_speed)

    if isinstance(speed_data, list):
        speed = int(speed_data[0].split()[0])
    elif isinstance(speed_data, str):
        if 'mph' in speed_data:
            speed = int(speed_data.split(' ')[0]) * 1.60934
        else:
            speed = int(speed_data.split()[0])
    elif isinstance(speed_data, (int, float)):
        speed = speed_data
    else:
        speed = default_speed

    return speed


def initialize_edge_usage(G):
    #Initialize or reset 'algorithm_uses' attribute for all edges to 0.
    nx.set_edge_attributes(G, 0, 'algorithm_uses')

def update_edge_usage(G, pred):
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
                        # For MultiGraphs, increment all edges between prev and target
                        for key in G[prev][target]:
                            if 'algorithm_uses' in G[prev][target][key]:
                                G[prev][target][key]['algorithm_uses'] += 1
                            else:
                                G[prev][target][key]['algorithm_uses'] = 1
                    else:
                        # For simple Graphs
                        if 'algorithm_uses' in G[prev][target]:
                            G[prev][target]['algorithm_uses'] += 1
                        else:
                            G[prev][target]['algorithm_uses'] = 1
                target = prev
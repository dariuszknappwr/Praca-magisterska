import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import random
import os
import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.colors as mcolors
import time
from pymongo import MongoClient
from graph_utils import set_speed_weigths
from bellman_ford import bellman_ford
from spfa import spfa
from spfa import initialize_spfa_edge_usage
from bellman_ford import initialize_bellman_ford_edge_usage
import psutil
from floyd_warshall import floyd_warshall
from johnson import johnson
from test_map import get_test_map
from generate_random_nodes import get_start_end_nodes
from tests_one_one import tests_one_one
from tests_one_many import tests_one_many
from tests_many_many import tests_many_many
from tests_one_one_k import tests_one_one_k



def main():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['PracaMagisterska']
    G_many_many_all_algorithm_uses = None
    for test_number in ['Test20']:
        #test_number = 'Test11'
        number_of_pairs = 100
        G = get_test_map(test_number)
        G = set_speed_weigths(G)
        print("Ustawiono wagi grafu")

        collection = db[test_number]

        #print(ox.basic_stats(G)) 
        
        start_nodes, end_nodes = get_start_end_nodes(test_number, number_of_pairs)
        for i in range(number_of_pairs):
            start = start_nodes[i]
            end = end_nodes[i]
            result = {}
            if test_number in ['Test1', 'Test2', 'Test3', 'Test4', 'Test5']:
                result = tests_one_one(G, start, end)
            elif test_number in ['Test6', 'Test7', 'Test8', 'Test9', 'Test10']:
                result = tests_one_many(G, start, end, plot=False)
            elif test_number in ['Test11', 'Test12', 'Test13', 'Test14', 'Test15', 'Test16', 'Test17']:
                result = tests_many_many(G, test_number, i, plot=True)
            elif test_number in ['Test18', 'Test19', 'Test20', 'Test21', 'Test22', 'Test23']:
                result = tests_one_one_k(G, start, end)

            collection.insert_one(result)

if __name__ == '__main__':
    main()
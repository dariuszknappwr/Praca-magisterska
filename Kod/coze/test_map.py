import osmnx as ox
import os

def load_local_map(osm_file_path):
    # Check if the file exists
    if os.path.isfile(osm_file_path):
        G = ox.load_graphml(osm_file_path)
        return G
    else:
        raise FileNotFoundError(f"OSM file cannot be found: {osm_file_path}")

def get_test_map(test_number):
    # Specify the path to your local .osm file
    test_map = {
        'Test1': 'Maps/Zalipie_map.graphml',
        'Test2': 'Maps/Wroclaw_map.graphml',
        'Test3': 'Maps/Lubelskie_map.graphml',
        'Test4': 'Maps/Berlin_map.graphml',
        'Test5': 'Maps/Nowy_York_map.graphml',
        'Test6': 'Maps/Zalipie_map.graphml',
        'Test7': 'Maps/Wroclaw_map.graphml',
        'Test8': 'Maps/Berlin_map.graphml',
        'Test9': 'Maps/Lubelskie_map.graphml',
        'Test10': 'Maps/Nowy_York_map.graphml',
        'Test11': 'Maps/Zalipie_map.graphml',
        'Test12': 'Maps/Tychowo_map.graphml',
        'Test13': 'Maps/Miloslaw_map.graphml',
        'Test14': 'Maps/Nowa_Deba_map.graphml',
        'Test15': 'Maps/Wadowice_map.graphml',
        'Test16': 'Maps/Plonsk_map.graphml',
        'Test17': 'Maps/Tczew_map.graphml',
        'Test18': 'Maps/Zalipie_map.graphml',
        'Test19': 'Maps/Tychowo_map.graphml',
        'Test20': 'Maps/Miloslaw_map.graphml',
        'Test21': 'Maps/Wadowice_map.graphml',
        'Test22': 'Maps/Tczew_map.graphml',
        'Test23': 'Maps/Ostrowiec_map.graphml',
        'Test24': 'Maps/Wroclaw_map.graphml'
    }

    local_osm_file_path = test_map.get(test_number)

    if local_osm_file_path is None:
        print(f"Invalid test number: {test_number}")
        return

    try:
        G = load_local_map(local_osm_file_path)
    except FileNotFoundError as e:
        print(e)
        return

    return G
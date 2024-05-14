import networkx as nx

def yen_ksp(G, source, target, K=1):
    A = [nx.shortest_path(G, source, target, weight='length')]
    B = []

    for k in range(1, K):
        for i in range(len(A[k - 1]) - 1):
            # Węzeł przypinający i ścieżka główna
            spur_node = A[k - 1][i]
            root_path = A[k - 1][:i + 1]

            # Usunięte krawędzie, które zostaną później przywrócone
            edges_removed = []

            # Usuwanie krawędzi zaangażowanych w dotychczas znalezione najkrótsze ścieżki
            for path in A:
                if len(path) > i and root_path == path[:i + 1]:
                    u, v = path[i], path[i + 1]
                    edge_keys = list(G[u][v].keys())  # Tworzenie osobnej listy kluczy krawędzi
                    for key in edge_keys:
                        edge_data = G[u][v][key]
                        G.remove_edge(u, v, key)
                        edges_removed.append((u, v, key, edge_data))

            spur_path = None
            try:
                # Obliczanie ścieżki od węzła przypinającego do celu
                spur_path = nx.shortest_path(G, spur_node, target, weight='length')
            except nx.NetworkXNoPath:
                pass

            # Ostateczna ścieżka to połączenie ścieżki głównej z odnalezioną ścieżką pomocniczą
            if spur_path is not None:
                total_path = root_path[:-1] + spur_path
                B.append(total_path)

            # Przywracanie usuniętych krawędzi
            for u, v, key, data in edges_removed:
                G.add_edge(u, v, key=key, **data)

        if not B:
            break  # Jeśli nie znaleziono żadnych ścieżek pomocniczych, wyszukiwanie zostaje zakończone

        # Posortowanie potencjalnych k najkrótszych ścieżek według ich długości
        B.sort(key=lambda path: sum(G[path[j]][path[j + 1]][0]['length'] for j in range(len(path) - 1)))
        # Dodanie najkrótszej ścieżki z lsity ścieżek pomocniczych do listy k najkrótszych ścieżek
        A.append(B[0])
        # Usunięcie najkrótszej ścieżki dodanej do A z listy B
        B.pop(0)

    return A
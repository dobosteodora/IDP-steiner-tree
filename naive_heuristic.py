import math
import random
import networkx as nx

from clustering import nearest_terminal_clustering


# implements a very naive heuristic; there is no guarantee that the resulting steiner tree is valid


def naive_heuristic(graph, terminals):
    steiner_tree = nx.Graph()

    clusters = nearest_terminal_clustering(graph, terminals)
    representatives = dict()  # key = representative, value = cluster
    list_representatives = []

    shortest_paths = dict()  # key = representative, value = list with dicts: key = representative, SP to representative
    for key in clusters:
        representative = random.choice(clusters[key])
        shortest_paths[representative] = []
        list_representatives.append(representative)
        representatives[representative] = key

    # compute shortest distances between cluster representatives
    for r1 in list_representatives:
        for r2 in list_representatives:
            if r1 < r2:
                length, path = nx.single_source_dijkstra(graph, source=representatives[r1],
                                                         target=representatives[r2],
                                                         weight='weight')
                shortest_paths[r1].append(
                    {'cluster': r2, 'distance': length, 'path': path})
                shortest_paths[r2].append(
                    {'cluster': r1, 'distance': length, 'path': path})

    count_terminals = len(terminals)
    count_processed_terminals = 1
    current_representative = list_representatives[0]
    covered_representatives = [list_representatives[0]]

    length, shortest_path = nx.single_source_dijkstra(graph, source=current_representative,
                                                      target=representatives[current_representative],
                                                      weight='weight')
    edges = get_edges_from_path(graph, shortest_path)
    for e in edges:
        steiner_tree.add_edge(e[0], e[1], weight=e[2])

    # connect representatives: choose the closest uncovered representative to the current tree
    while count_processed_terminals < count_terminals:
        global_min_dist = math.inf
        global_closest_repr = None
        for r in covered_representatives:
            sp = shortest_paths[r]  # list of dicts
            closest_repr = min(sp,
                               key=lambda x: x['distance'] if x['cluster'] not in covered_representatives else math.inf)
            dist = closest_repr['distance']
            if dist < global_min_dist:
                global_closest_repr = closest_repr['cluster']
                global_min_dist = dist
                current_representative = r

        length, shortest_path = nx.single_source_dijkstra(graph, source=current_representative,
                                                          target=global_closest_repr,
                                                          weight='weight')
        covered_representatives.append(global_closest_repr)
        edges = get_edges_from_path(graph, shortest_path)
        for e in edges:
            steiner_tree.add_edge(e[0], e[1], weight=e[2])

        count_processed_terminals += 1

        length, shortest_path = nx.single_source_dijkstra(graph, source=global_closest_repr,
                                                          target=representatives[global_closest_repr],
                                                          weight='weight')
        edges = get_edges_from_path(graph, shortest_path)
        for e in edges:
            steiner_tree.add_edge(e[0], e[1], weight=e[2])

    weight = 0
    for u, v in steiner_tree.edges:
        weight += steiner_tree[u][v]['weight']

    return steiner_tree, weight


def get_edges_from_path(graph, path):
    edges = []
    index = 0
    count_nodes_in_path = len(path)
    while index < count_nodes_in_path - 1:
        u = path[index]
        v = path[index + 1]
        edges.append((u, v, graph[u][v]['weight']))
        index += 1
    return edges

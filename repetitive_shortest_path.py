import math
import networkx as nx


def repetitive_shortest_path(graph, terminals):
    # compute shortest paths from terminals to all nodes
    shortest_paths = dict()  # key = terminal, values = length, path for the shortest paths to all nodes
    for terminal in terminals:
        length, path = nx.single_source_dijkstra(graph, terminal, weight='weight')
        shortest_paths[terminal] = {'length': length, 'path': path}

    # ensure that the stored SP from terminal1 to terminal2 is the same as the SP from terminal2 to terminal1
    # in case of multiple shortest paths
    for t1 in terminals:
        for t2 in terminals:
            if t1 < t2:
                path = shortest_paths[t1].get('path')[t2]
                shortest_paths[t2].get('path')[t1] = path

    count_processed_terminals = 1
    covered_terminals = [terminals[0]]

    steiner_tree = nx.Graph()
    count_terminals = len(terminals)

    while count_processed_terminals < count_terminals:
        # get the closest uncovered terminal to the covered terminals
        global_min_dist_to_uncovered_terminal = math.inf
        global_closest_uncovered_terminal = None
        current_terminal = None

        for terminal in covered_terminals:
            length = shortest_paths[terminal].get('length')
            uncovered_terminals = [(x, length[x]) for x in length if x not in covered_terminals and x != terminal
                                   and x in terminals]

            closest_uncovered_terminal = min(uncovered_terminals, key=lambda x: x[1])[0]
            min_dist_to_uncovered_terminal = min(uncovered_terminals, key=lambda x: x[1])[1]

            if min_dist_to_uncovered_terminal < global_min_dist_to_uncovered_terminal:
                current_terminal = terminal
                global_min_dist_to_uncovered_terminal = min_dist_to_uncovered_terminal
                global_closest_uncovered_terminal = closest_uncovered_terminal

        # add the edges on the shortest path to the chosen uncovered terminal to the current tree
        index = 0
        shortest_path = shortest_paths[current_terminal].get('path')[global_closest_uncovered_terminal]
        count_nodes_in_path = len(shortest_path)
        while index < count_nodes_in_path - 1:
            u = shortest_path[index]
            v = shortest_path[index + 1]
            steiner_tree.add_edge(u, v, weight=graph[u][v]['weight'])
            index += 1

        count_processed_terminals += 1
        covered_terminals.append(global_closest_uncovered_terminal)

    # successively remove non-terminals with degree 1
    done = False
    while not done:
        done = True
        for node in list(steiner_tree.nodes):
            if node not in terminals and steiner_tree.degree[node] == 1:
                done = False
                steiner_tree.remove_node(node)

    weight = 0
    for u, v in steiner_tree.edges:
        weight += steiner_tree[u][v]['weight']

    return steiner_tree, weight

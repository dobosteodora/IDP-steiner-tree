import networkx as nx


# implements the algorithm described here: https://people.mpi-inf.mpg.de/~mehlhorn/ftp/SteinerTrees.pdf
def mehlhorn_algorithm(graph, terminals):
    # Step 1: create graph G1'
    # add a new vertex s0 to the input graph and connect it to all the terminals with cost 0
    s0 = len(graph.nodes) + 1
    G1_prime = nx.Graph()

    terminals_pairs = ((t1, t2) for t1 in terminals for t2 in terminals if t2 > t1)
    d1_prime = dict()
    for (t1, t2) in terminals_pairs:
        d1_prime[(t1, t2)] = {'distance': None, 'node 1': None, 'node 2': None}

    for t in terminals:
        graph.add_edge(s0, t, weight=0)

    length_shortest_paths, shortest_paths = nx.single_source_dijkstra(graph, s0)

    # create, for each terminal t, its voronoi region, which contains the nodes that are closest to t
    # and their distance to t in the initial graph
    voronoi_regions = dict()
    for t in terminals:
        voronoi_regions[t] = dict()
        voronoi_regions[t][t] = 0

    closest_terminal = dict()

    # add nodes and their distance to the voronoi regions
    for element in shortest_paths:
        current_node = shortest_paths[element][len(shortest_paths[element]) - 1]
        path = []
        for node in reversed(shortest_paths[element]):
            path.append(node)
            if node in terminals:
                voronoi_regions[node][current_node] = length_shortest_paths[current_node]
                closest_terminal[current_node] = {'terminal': node, 'distance': length_shortest_paths[current_node],
                                                  'path': path}
                break

    for (u, v) in graph.edges:
        if u != s0 and v != s0 and closest_terminal[u]['terminal'] != closest_terminal[v]['terminal']:
            # update distance between two terminals in d1_prime if applicable
            if closest_terminal[u]['terminal'] < closest_terminal[v]['terminal']:
                t1 = closest_terminal[u]['terminal']
                t2 = closest_terminal[v]['terminal']
            else:
                t1 = closest_terminal[v]['terminal']
                t2 = closest_terminal[u]['terminal']

            if d1_prime[(t1, t2)]['distance'] is None or d1_prime[(t1, t2)]['distance'] > closest_terminal[u][
                'distance'] + graph[u][v]['weight'] + closest_terminal[v]['distance']:
                d1_prime[(t1, t2)] = {
                    'distance': closest_terminal[u]['distance'] + graph[u][v]['weight'] + closest_terminal[v][
                        'distance'],
                    'node 1': u, 'node 2': v}

    # add edges in G1_prime
    for (t1, t2) in d1_prime.keys():
        # print(d1_prime[(t1, t2)]['distance'])
        if d1_prime[(t1, t2)]['distance'] is not None:
            G1_prime.add_edge(t1, t2, weight=d1_prime[(t1, t2)]['distance'])

    # Step 2: compute MST of G1_prime
    G2 = nx.minimum_spanning_tree(G1_prime, 'weight', 'prim')

    # Step 3: replace each edge in G2 by its corresponding shortest path in the initial graph
    G3 = nx.Graph()
    for (t1, t2) in G2.edges:
        if t1 > t2:
            tmp = t1
            t1 = t2
            t2 = tmp
        node_1 = d1_prime[(t1, t2)]['node 1']
        node_2 = d1_prime[(t1, t2)]['node 2']

        G3.add_edge(node_1, node_2, weight=graph[node_1][node_2]['weight'])

        paths = [closest_terminal[node_1]['path'], closest_terminal[node_2]['path']]
        for path in paths:
            index = 0
            count_nodes_in_path = len(path)
            while index < count_nodes_in_path - 1:
                u = path[index]
                v = path[index + 1]
                G3.add_edge(u, v, weight=graph[u][v]['weight'])
                index += 1

    # Step 4: compute MST of G3
    G4 = nx.minimum_spanning_tree(G3, 'weight', 'prim')

    # Step 5: construct Steiner tree G5 by deleting edges in G4, if necessary,
    # so that no leaves in G5 are Steiner vertices
    done = False
    while done is not True:
        done = True
        for node in G4.nodes:
            if G4.degree[node] == 1 and node not in terminals:
                G4.remove_node(node)
                done = False

    weight = 0
    for (u, v) in G4.edges:
        weight += G4[u][v]['weight']
        
    return G4, weight

import networkx as nx
import numpy as np

# global variables that will not change during the execution of the algorithm
graph = nx.Graph()

# encoding of the nodes in a suitable format for the algorithm
nodes = dict()


# sort edges w.r.t. growth value
# growth = the "amount" that an edge needs to be tight
def sort_edges(edges):
    for edge in edges:
        s = edges[edge]['start_node']
        e = edges[edge]['end_node']
        g = edges[edge]['weight'] - (nodes[s]['dual'] + nodes[e]['dual'])

        if nodes[s]['is_active'] is True and nodes[e]['is_active'] is True:
            g /= 2

        edges[edge]['growth'] = g

    return list(sorted(edges, key=lambda ed: edges[ed]['growth']))


def iteration(sorted_edges, incident_edges, steiner_tree, covered_edges, count_active_sets, x_graph):
    tight_edge = sorted_edges[0]

    # print current tight edge for debugging
    # print(tight_edge)

    s = incident_edges[tight_edge]['start_node']
    e = incident_edges[tight_edge]['end_node']

    g = incident_edges[tight_edge]['growth']

    # update dual variable for each active node
    for n in nodes:
        if nodes[n]['is_active'] is True:
            nodes[n]['dual'] += g

    # add new incident edges to the set of considered edges
    set_edges = list(graph.edges(s)) + list(graph.edges(e))
    for edge in set_edges:
        first = edge[0]
        second = edge[1]
        if (first, second) not in covered_edges and (second, first) not in covered_edges:
            incident_edges[edge] = {'start_node': first, 'end_node': second,
                                    'weight': graph[first][second]['weight'],
                                    'growth': graph[first][second]['weight'] -
                                              (nodes[first]['dual'] + nodes[second]['dual'])
                                    }
            if nodes[first]['is_active'] is True and nodes[second]['is_active'] is True:
                incident_edges[edge]['growth'] = incident_edges[edge]['growth'] / 2

    covered_edges.add(tight_edge)

    nodes[s]['is_active'] = True
    nodes[e]['is_active'] = True

    # CASE 1: nodes are in the same active set: nothing to update
    if nodes[s]['active_set'] == nodes[e]['active_set']:
        pass

    # CASE 2: s is non-terminal and e is terminal: add s to the active set of e
    elif len(nodes[s]['active_set']) == 0 and len(nodes[e]['active_set']) != 0:

        x_graph.add_edge(s, e, weight=incident_edges[tight_edge]['weight'])

        as_prev = nodes[e]['active_set']
        nodes[e]['active_set'].add(s)

        for n in as_prev:
            nodes[n]['active_set'] = nodes[e]['active_set']

    # CASE 3: e is non-terminal and s is terminal: add e to the active set of s
    elif len(nodes[e]['active_set']) == 0 and len(nodes[s]['active_set']) != 0:

        x_graph.add_edge(s, e, weight=incident_edges[tight_edge]['weight'])

        as_prev = nodes[s]['active_set']
        nodes[s]['active_set'].add(e)

        for n in as_prev:
            nodes[n]['active_set'] = nodes[s]['active_set']

    # CASE 4: both s and e are in active sets -> merge active sets, find connecting path to update the steiner tree
    # the connecting path must consist of edges which have primal value 1
    elif len(nodes[e]['active_set']) != 0 and len(nodes[s]['active_set']) != 0:

        x_graph.add_edge(s, e, weight=incident_edges[tight_edge]['weight'])

        terminals_comp1 = list(filter(lambda node: nodes[node]['is_terminal'] is True, nodes[e]['active_set']))
        terminals_comp2 = list(filter(lambda node: nodes[node]['is_terminal'] is True, nodes[s]['active_set']))

        t_comp1 = terminals_comp1[0]
        t_comp2 = terminals_comp2[0]

        # find path between 2 terminals, one from one component and the other one from the other component
        # with modified DFS (only one path exists, but the method "searches" for all such paths...)
        path_generator = nx.all_simple_paths(x_graph, source=t_comp1, target=t_comp2)
        for path in path_generator:
            index = 0
            count_nodes_in_path = len(path)
            while index < count_nodes_in_path - 1:
                u = path[index]
                v = path[index + 1]
                steiner_tree.add_edge(u, v, weight=x_graph[u][v]['weight'])
                index += 1

        # alternative: find shortest path between terminals - same result as above...
        # path = nx.dijkstra_path(x_graph, t_comp1, t_comp2)
        # index = 0
        # count_nodes_in_path = len(path)
        # while index < count_nodes_in_path - 1:
        #     u = path[index]
        #     v = path[index + 1]
        #     steiner_tree.add_edge(u, v, weight=x_graph[u][v]['weight'])
        #     index += 1

        if nodes[e]['active_set'] != nodes[s]['active_set']:
            new_active_set = nodes[e]['active_set']
            new_active_set.update(nodes[s]['active_set'])

            for n in new_active_set:
                nodes[n]['active_set'] = new_active_set

        # since we merged two active sets, we have one less active set
        count_active_sets -= 1

    # delete the current tight edge from the set of incident edges, since we will never consider it again
    del incident_edges[tight_edge]

    return sorted_edges, incident_edges, steiner_tree, covered_edges, count_active_sets, x_graph


def primal_dual(g, terminals):
    global graph
    graph = g

    steiner_tree = nx.Graph()
    steiner_tree.add_nodes_from(terminals)

    count_active_sets = 0

    # graph that contains edges with primal value of 1
    x_graph = nx.Graph()

    for node in graph.nodes:
        global nodes
        nodes[node] = {'dual': 0, 'is_terminal': False, 'is_active': False, 'active_set': set()}
        if node in terminals:
            nodes[node]['is_terminal'] = True
            nodes[node]['is_active'] = True
            nodes[node]['active_set'].add(node)
            count_active_sets += 1

    # dictionary of edges that are incident to at least one active node (tight edges are always chosen from this dict)
    incident_edges = dict()
    for t in terminals:
        terminal_edges = graph.edges(t)
        for e in terminal_edges:
            incident_edges[e] = {'start_node': e[0], 'end_node': e[1],
                                 'weight': graph[e[0]][e[1]]['weight'], 'growth': graph[e[0]][e[1]]['weight']}

    # set of edges that are tight
    covered_edges = set()

    # early_stopping = 0.7  # early stopping w.p. 1 - 0.7 -> set to 1 if no early stopping allowed
    # stop = 0
    #
    # it = 1
    # if len(terminals) >= 2:
    #     min_it = 2
    # else:
    #     min_it = 1

    while count_active_sets > 1:
        # print("Iteration {}".format(it))
        # it += 1
        sorted_edges = sort_edges(incident_edges)
        sorted_edges, incident_edges, steiner_tree, covered_edges, count_active_sets, x_graph = \
            iteration(sorted_edges, incident_edges, steiner_tree, covered_edges, count_active_sets, x_graph)
        # if count_active_sets < len(terminals) - min_it:
        #     stop = np.random.uniform(0, 1)
        #     print("Random number: {}".format(stop))

    # weight of the solution before early stopping
    weight = 0
    for u, v in steiner_tree.edges:
        weight += steiner_tree[u][v]['weight']

    # early stopping
    # if count_active_sets > 1 and stop >= early_stopping:
    #     count_connected_comps = nx.number_connected_components(steiner_tree)
    #     print("Early stopping when there are {} active sets".format(count_active_sets))
    #     print("Current cost: {}, number components: {}".format(weight, count_connected_comps))
    #     print("Cost per terminal: {}".format(weight / len(terminals)))
    #
    #     new_edges = nx.Graph()
    #
    #     # connect the two largest connected components
    #     while count_connected_comps > 1:
    #         c1 = [c for c in sorted(nx.connected_components(steiner_tree), key=len, reverse=False)][
    #             len(list(nx.connected_components(steiner_tree))) - 1]
    #         c2 = [c for c in sorted(nx.connected_components(steiner_tree), key=len, reverse=False)][
    #             len(list(nx.connected_components(steiner_tree))) - 2]
    #
    #         print("Merge components {} and {}".format(c1, c2))
    #
    #         terminals1 = list(filter(lambda n: n in terminals, list(c1)))
    #         t1 = terminals1[0]
    #         terminals2 = list(filter(lambda n: n in terminals, list(c2)))
    #         t2 = terminals2[0]
    #         shortest_path = nx.dijkstra_path(g, t1, t2, weight='weight')
    #
    #         index = 0
    #         count_nodes_in_path = len(shortest_path)
    #         while index < count_nodes_in_path - 1:
    #             u = shortest_path[index]
    #             v = shortest_path[index + 1]
    #             steiner_tree.add_edge(u, v, weight=g[u][v]['weight'])
    #             new_edges.add_edge(u, v, weight=g[u][v]['weight'])
    #             index += 1
    #
    #         count_connected_comps = nx.number_connected_components(steiner_tree)
    #
    #     subtract_cost = 0
    #     while not nx.is_tree(steiner_tree):
    #         edges_cycle = list(nx.find_cycle(steiner_tree))
    #         for (u, v) in edges_cycle:
    #             subtract_cost += steiner_tree[u][v]['weight']
    #             steiner_tree.remove_edge(u, v)
    #             break
    #
    #     additional_cost = 0
    #     for u, v in new_edges.edges:
    #         additional_cost += new_edges[u][v]['weight']
    #
    #     additional_cost -= subtract_cost
    #
    #     print("Additional cost: {}".format(additional_cost))
    #     print("Additional cost per terminal: {}".format(additional_cost / len(terminals)))
    #
    #     weight += additional_cost

    return steiner_tree, weight

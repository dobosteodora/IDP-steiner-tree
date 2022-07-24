import sys
import networkx as nx
import numpy as np

from reinforcement_learning import Agent, dqn
from naive_heuristic import naive_heuristic
from primal_dual import primal_dual
from repetitive_shortest_path import repetitive_shortest_path
from mehlhorn_algorithm import mehlhorn_algorithm


# create graph given data in PACE-challenge format: https://pacechallenge.org/2018/steiner-tree/
def create_graph_from_input():
    sys.stdin.readline()
    n = int(sys.stdin.readline().split()[1])
    m = int(sys.stdin.readline().split()[1])

    graph = nx.Graph()
    graph.add_nodes_from(list(range(1, n + 1)))

    # add edges
    for edge in range(0, m):
        line = sys.stdin.readline()
        tokens = line.split()
        source = int(tokens[1])
        target = int(tokens[2])
        weight = int(tokens[3])
        graph.add_edge(source, target)
        graph[source][target]['weight'] = weight
        graph[source][target]['similarity'] = 1 / weight  # use for kMeans

    for i in range(0, 3):
        sys.stdin.readline()

    line = sys.stdin.readline()
    t = int(line.split()[1])
    terminals = []

    for terminal in range(0, t):
        line = sys.stdin.readline()
        tokens = line.split()
        graph.nodes[int(tokens[1])]['terminal'] = True
        terminals.append(int(tokens[1]))

    return graph, terminals


# checks if a steiner tree is valid
def valid_solution(graph, terminals):
    is_connected = nx.is_connected(graph)
    if not is_connected:
        print("Graph is not connected")
    is_tree = nx.is_tree(graph)
    if not is_tree:
        print("Graph is not a tree")
    all_terminals_covered = all(node in graph.nodes for node in terminals)
    if not all_terminals_covered:
        print("There exist uncovered terminals")
    return is_connected and is_tree and all_terminals_covered


g, t = create_graph_from_input()


# print("Computing steiner tree using nx library...")
# steiner_tree = nx.algorithms.approximation.steinertree.steiner_tree(g, t)
# print("Weight: " + steiner_tree.size(weight="weight").__str__())
# print("Solution is valid: {}\n".format(valid_solution(steiner_tree, t)))


# print("Running repetitive shortest path heuristic...")
# steiner_tree, weight = repetitive_shortest_path(g, t)
# print("Weight: {}".format(weight))
# print("Solution is valid: {}\n".format(valid_solution(steiner_tree, t)))


# print("Running primal-dual algorithm...")
# steiner_tree, weight = primal_dual(g, t)
# print("Weight: {}".format(weight))
# print("Solution is valid: {}\n".format(valid_solution(steiner_tree, t)))


print("Running mehlhorn algorithm...")
steiner_tree, weight = mehlhorn_algorithm(g, t)
print("Weight: {}".format(weight))
print("Solution is valid: {}\n".format(valid_solution(steiner_tree, t)))


# print("Running Cherrypick...")
# steiner_tree, weight = dqn(g, t, len(g.nodes))
# print("Weight: {}".format(weight))
# print("Solution is valid: {}\n".format(valid_solution(steiner_tree, t)))


import networkx as nx
import random
import matplotlib.pyplot as plt

# generate graphs: complete, ladder, wheel, grid

file_no = 1


def plot_graph(graph, name):
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    # plt.show()
    plt.savefig(name)
    plt.clf()


# generate graphs
def assign_random_weights(graph, lower_bound, upper_bound):
    for (u, v) in graph.edges():
        graph.edges[u, v]['weight'] = random.randint(lower_bound, upper_bound)


# create a random graph

# seed = 20160
# G_erdos = nx.erdos_renyi_graph(10, 0.5, seed=seed)
# if nx.is_connected(G_erdos):
#     assign_random_weights(G_erdos, 1, 100)


# complete graphs

# with 5 nodes
complete_5 = nx.complete_graph(5)
assign_random_weights(complete_5, 1, 50)
complete_5_terminals_count = random.randint(2, 5)
terminals_complete_5 = random.sample(complete_5.nodes(), complete_5_terminals_count)


file = open("graph{}.txt".format(file_no), "w")

file.write("Complete 5\n")
file.write("SECTION Graph\n")
file.write("Nodes {}\n".format(len(complete_5.nodes())))
file.write("Edges {}\n".format(len(complete_5.edges())))
for (u,v) in complete_5.edges():
    file.write("E {} {} {}\n".format(u, v, complete_5.edges[u, v]['weight']))
file.write("END\n\n")
file.write("SECTION Terminals\n")
file.write("Terminals {}\n".format(complete_5_terminals_count))
for t in terminals_complete_5:
    file.write("T {}\n".format(t))
file.write("END\n\n")
file.write("EOF")
file.close()
plot_graph(complete_5, 'complete_5.png')

file_no += 1


# with 10 nodes
complete_10 = nx.complete_graph(10)
assign_random_weights(complete_10, 1, 50)
complete_10_terminals_count = random.randint(2, 7)
terminals_complete_10 = random.sample(complete_10.nodes(), complete_10_terminals_count)


file = open("graph{}.txt".format(file_no), "x")

file.write("Complete 10\n")
file.write("SECTION Graph\n")
file.write("Nodes {}\n".format(len(complete_10.nodes())))
file.write("Edges {}\n".format(len(complete_10.edges())))
for (u,v) in complete_10.edges():
    file.write("E {} {} {}\n".format(u, v, complete_10.edges[u, v]['weight']))
file.write("END\n\n")
file.write("SECTION Terminals\n")
file.write("Terminals {}\n".format(complete_10_terminals_count))
for t in terminals_complete_10:
    file.write("T {}\n".format(t))
file.write("END\n\n")
file.write("EOF")
file.close()
plot_graph(complete_10, 'complete_10.png')

file_no += 1

# with 20 nodes
complete_20 = nx.complete_graph(20)
assign_random_weights(complete_20, 1, 50)
complete_20_terminals_count = random.randint(2, 15)
terminals_complete_20 = random.sample(complete_20.nodes(), complete_20_terminals_count)


file = open("graph{}.txt".format(file_no), "x")

file.write("Complete 20\n")
file.write("SECTION Graph\n")
file.write("Nodes {}\n".format(len(complete_20.nodes())))
file.write("Edges {}\n".format(len(complete_20.edges())))
for (u,v) in complete_20.edges():
    file.write("E {} {} {}\n".format(u, v, complete_20.edges[u, v]['weight']))
file.write("END\n\n")
file.write("SECTION Terminals\n")
file.write("Terminals {}\n".format(complete_20_terminals_count))
for t in terminals_complete_20:
    file.write("T {}\n".format(t))
file.write("END\n\n")
file.write("EOF")
file.close()
plot_graph(complete_20, 'complete_20.png')

file_no += 1


# ladder graph

# with n = 5
ladder_5 = nx.ladder_graph(5)
assign_random_weights(ladder_5, 1, 50)
ladder_5_terminals_count = random.randint(2, 9)
terminals_ladder_5 = random.sample(ladder_5.nodes(), ladder_5_terminals_count)


file = open("graph{}.txt".format(file_no), "x")

file.write("Ladder 5\n")
file.write("SECTION Graph\n")
file.write("Nodes {}\n".format(len(ladder_5.nodes())))
file.write("Edges {}\n".format(len(ladder_5.edges())))
for (u,v) in ladder_5.edges():
    file.write("E {} {} {}\n".format(u, v, ladder_5.edges[u, v]['weight']))
file.write("END\n\n")
file.write("SECTION Terminals\n")
file.write("Terminals {}\n".format(ladder_5_terminals_count))
for t in terminals_ladder_5:
    file.write("T {}\n".format(t))
file.write("END\n\n")
file.write("EOF")
file.close()
plot_graph(ladder_5, 'ladder_5.png')

file_no += 1


# with n = 10
ladder_10 = nx.ladder_graph(10)
assign_random_weights(ladder_10, 1, 50)
ladder_10_terminals_count = random.randint(2, 15)
terminals_ladder_10 = random.sample(ladder_10.nodes(), ladder_10_terminals_count)


file = open("graph{}.txt".format(file_no), "x")

file.write("Ladder 10\n")
file.write("SECTION Graph\n")
file.write("Nodes {}\n".format(len(ladder_10.nodes())))
file.write("Edges {}\n".format(len(ladder_10.edges())))
for (u,v) in ladder_10.edges():
    file.write("E {} {} {}\n".format(u, v, ladder_10.edges[u, v]['weight']))
file.write("END\n\n")
file.write("SECTION Terminals\n")
file.write("Terminals {}\n".format(ladder_10_terminals_count))
for t in terminals_ladder_10:
    file.write("T {}\n".format(t))
file.write("END\n\n")
file.write("EOF")
file.close()
plot_graph(ladder_10, 'ladder_10.png')

file_no += 1


# wheel graphs

# with 10 nodes
wheel_10 = nx.wheel_graph(10)
assign_random_weights(wheel_10, 1, 50)
wheel_10_terminals_count = random.randint(2, 7)
terminals_wheel_10 = random.sample(wheel_10.nodes(), wheel_10_terminals_count)


file = open("graph{}.txt".format(file_no), "x")

file.write("Wheel 10\n")
file.write("SECTION Graph\n")
file.write("Nodes {}\n".format(len(wheel_10.nodes())))
file.write("Edges {}\n".format(len(wheel_10.edges())))
for (u,v) in wheel_10.edges():
    file.write("E {} {} {}\n".format(u, v, wheel_10.edges[u, v]['weight']))
file.write("END\n\n")
file.write("SECTION Terminals\n")
file.write("Terminals {}\n".format(wheel_10_terminals_count))
for t in terminals_wheel_10:
    file.write("T {}\n".format(t))
file.write("END\n\n")
file.write("EOF")
file.close()
plot_graph(wheel_10, 'wheel_10.png')

file_no += 1

# with n = 20 nodes
wheel_20 = nx.wheel_graph(20)
assign_random_weights(wheel_20, 1, 50)
wheel_20_terminals_count = random.randint(2, 15)
terminals_wheel_20 = random.sample(wheel_20.nodes(), wheel_20_terminals_count)


file = open("graph{}.txt".format(file_no), "x")

file.write("Wheel 20\n")
file.write("SECTION Graph\n")
file.write("Nodes {}\n".format(len(wheel_20.nodes())))
file.write("Edges {}\n".format(len(wheel_20.edges())))
for (u,v) in wheel_20.edges():
    file.write("E {} {} {}\n".format(u, v, wheel_20.edges[u, v]['weight']))
file.write("END\n\n")
file.write("SECTION Terminals\n")
file.write("Terminals {}\n".format(wheel_20_terminals_count))
for t in terminals_wheel_20:
    file.write("T {}\n".format(t))
file.write("END\n\n")
file.write("EOF")
file.close()
plot_graph(wheel_20, 'wheel_20.png')

file_no += 1


# grid graphs ---- not really suitable for our graph format since position of each node must be stored

# with 5 nodes
grid_5 = nx.grid_2d_graph(5, 4)
assign_random_weights(grid_5, 1, 50)
grid_5_terminals_count = random.randint(2, 4)
terminals_grid_5 = random.sample(grid_5.nodes(), grid_5_terminals_count)


file = open("graph{}.txt".format(file_no), "x")

file.write("Grid 10\n")
file.write("SECTION Graph\n")
file.write("Nodes {}\n".format(len(grid_5.nodes())))
file.write("Edges {}\n".format(len(grid_5.edges())))
for (u,v) in grid_5.edges():
    file.write("E {} {} {}\n".format(u, v, grid_5.edges[u, v]['weight']))
file.write("END\n\n")
file.write("SECTION Terminals\n")
file.write("Terminals {}\n".format(grid_5_terminals_count))
for t in terminals_grid_5:
    file.write("T {}\n".format(t))
file.write("END\n\n")
file.write("EOF")
file.close()
plot_graph(grid_5, 'grid_5.png')

file_no += 1


# with 10 nodes
grid_10 = nx.grid_2d_graph(10, 4)
assign_random_weights(grid_10, 1, 50)
grid_10_terminals_count = random.randint(2, 7)
terminals_grid_10 = random.sample(grid_10.nodes(), grid_10_terminals_count)


file = open("graph{}.txt".format(file_no), "x")

file.write("Grid 10\n")
file.write("SECTION Graph\n")
file.write("Nodes {}\n".format(len(grid_10.nodes())))
file.write("Edges {}\n".format(len(grid_10.edges())))
for (u,v) in grid_10.edges():
    file.write("E {} {} {}\n".format(u, v, grid_10.edges[u, v]['weight']))
file.write("END\n\n")
file.write("SECTION Terminals\n")
file.write("Terminals {}\n".format(grid_10_terminals_count))
for t in terminals_grid_10:
    file.write("T {}\n".format(t))
file.write("END\n\n")
file.write("EOF")
file.close()
plot_graph(grid_10, 'grid_10.png')

file_no += 1


# with 20 nodes
grid_20 = nx.grid_2d_graph(20, 4)
assign_random_weights(grid_20, 1, 50)
grid_20_terminals_count = random.randint(2, 15)
terminals_grid_20 = random.sample(grid_20.nodes(), grid_20_terminals_count)


file = open("graph{}.txt".format(file_no), "x")

file.write("Grid 20\n")
file.write("SECTION Graph\n")
file.write("Nodes {}\n".format(len(grid_20.nodes())))
file.write("Edges {}\n".format(len(grid_20.edges())))
for (u,v) in grid_20.edges():
    file.write("E {} {} {}\n".format(u, v, grid_20.edges[u, v]['weight']))
file.write("END\n\n")
file.write("SECTION Terminals\n")
file.write("Terminals {}\n".format(grid_20_terminals_count))
for t in terminals_grid_20:
    file.write("T {}\n".format(t))
file.write("END\n\n")
file.write("EOF")
file.close()
plot_graph(grid_20, 'grid_20.png')

file_no += 1


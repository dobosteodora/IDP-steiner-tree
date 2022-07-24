import math
import networkx as nx
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

# implements some clustering approaches for the nodes in a graph


def get_first_eigenvectors(graph):
    L_sparse = nx.linalg.laplacianmatrix.laplacian_matrix(graph, weight='similarity')
    L = sp.sparse.csr_matrix.todense(L_sparse)
    eigenvalues, eigenvectors = np.linalg.eig(np.asarray(L))

    # sort eigenvectors in increasing order w.r.t corresponding eigenvalues
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]
    return eigenvectors


# clustering based based on squared Euclidean distance to the terminals
def spectral_clustering(graph, terminals):
    eigenvectors = get_first_eigenvectors(graph)
    embeddings = eigenvectors[:, 1:len(terminals) + 1]

    clusters = dict()  # key = terminal, values = list with the nodes in the cluster represented by terminal
    for t in terminals:
        clusters[t] = [t]

    for node in graph.nodes:
        if node not in terminals:
            min_dist = math.inf
            c = None
            for terminal in terminals:
                dist = np.sum((embeddings[node - 1, :] - embeddings[terminal - 1, :]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    c = terminal
            clusters[c].append(node)

    return clusters


# creates #terminals clusters, where each cluster contains a terminal
# node is assigned to the cluster for which the shortest path to the corresponding terminal is minimal
def nearest_terminal_clustering(graph, terminals):
    clusters = dict()  # key = terminal, values = list with the nodes in the cluster represented by terminal
    shortest_paths = dict()  # key = terminal, values = length, path for the shortest paths to all nodes
    for t in terminals:
        clusters[t] = [t]
        length, path = nx.single_source_dijkstra(graph, t, weight='weight')
        shortest_paths[t] = {'length': length, 'path': path}

    for node in graph.nodes:
        if node not in terminals:
            dist_to_terminals = [(t, shortest_paths[t].get('length')[node]) for t in terminals]
            nearest_terminal = min(dist_to_terminals, key=lambda t: t[1])[0]
            clusters[nearest_terminal].append(node)

    return clusters


def kMeans_clustering(graph, terminals):
    eigenvectors = get_first_eigenvectors(graph)

    terminals_embeddings = []
    for t in terminals:
        terminals_embeddings.append(eigenvectors[t - 1, 1:len(terminals) + 1])

    terminals_embeddings = np.array(terminals_embeddings)

    # kMeans on #terminals eigenvectors with associated nonzero eigenvalues
    kMeans = KMeans(n_clusters=len(terminals), init=terminals_embeddings, n_init=1, max_iter=1)
    kMeans.fit(eigenvectors[:, 1:len(terminals) + 1])
    labels = kMeans.labels_

    return labels


# TODO make sure the terminals are in different clusters, initial centroids should be the terminals...
# DOES NOT WORK AT THE MOMENT
def kMedoids_clustering(graph, terminals):
    eigenvectors = get_first_eigenvectors(graph)

    terminals_embeddings = []
    for t in terminals:
        terminals_embeddings.append(eigenvectors[t - 1, 1:len(terminals) + 1])

    # kMedoids on #terminals eigenvectors with associated nonzero eigenvalues
    kMedoids = KMedoids(n_clusters=len(terminals), random_state=0)
    kMedoids.fit(eigenvectors[:, 1:len(terminals) + 1])
    labels = kMedoids.labels_

    return labels


# get clusters from a list of node-cluster assignments
def get_clusters(graph, labels, terminals):
    clusters = []
    for l in range(0, len(labels)):
        graph.nodes[l + 1]['cluster'] = labels[l]
    print(nx.get_node_attributes(graph, 'cluster'))

    for l in range(0, len(terminals)):
        cluster = [node for node, attr in graph.nodes(data=True) if attr['cluster'] == l]
        clusters.append(cluster)

    return clusters

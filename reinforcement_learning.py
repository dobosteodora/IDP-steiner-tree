import math
import networkx as nx
import numpy as np
from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import csv
from torch.autograd import Variable

initial_graph = nx.Graph()
terminals = []
s_v = np.array([])
t_v = np.array([])
matrix = np.matrix([])
neighborhoods = dict()
const = None
covered_terminals = []
uncovered_terminals = []

steiner_tree = nx.Graph()
parent = dict()

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 1
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 1e-5  # for SGD
MOMENTUM = 0.9  # for SGD
UPDATE_EVERY = 5  # how often to update the network

used_prev_weights = False  # change to True to use weights in weights.csv 
one_it = False


# preprocessing of the input graph - create similarity matrix
# first row will have infinity values on all columns in order to start indexing from 1
# (first node of the graph is always 1)
def compute_similarity_matrix(k=2):
    global matrix
    matrix = np.matrix(np.ones((len(initial_graph.nodes) + 1, k)) * np.inf)

    # compute the shortest paths to k closest terminals
    i = 0
    for node in initial_graph.nodes:
        i += 1
        j = 0
        distance = dict()
        # consider only the terminals which are not included in the solution
        for terminal in uncovered_terminals:
            if node == terminal:
                distance[terminal] = 0
            else:
                length = nx.shortest_path_length(initial_graph, source=node,
                                                 target=terminal,
                                                 weight='weight')
                distance[terminal] = length
        # add the row corresponding to the current node to the matrix
        for entry in range(k):
            if len(distance) > 0:
                tmp = min(distance, key=distance.get)
                if distance.get(tmp) != 0:  # compute similarity
                    matrix[i, j] = 1 / distance.get(tmp)
                else:
                    matrix[i, j] = 0
                j += 1
                del distance[tmp]
            else:
                matrix[i, j] = 0
                j += 1


# get set of neighbors for all nodes
def compute_neighborhoods():
    global neighborhoods
    for node in initial_graph.nodes:
        iterator = initial_graph.neighbors(node)
        list_neighbors = []
        for n in iterator:
            list_neighbors.append(n)
        neighborhoods[node] = list_neighbors
    return neighborhoods


# concatenate s and t values for a node
def connection_op(node):
    c = torch.Tensor([s_v[node], t_v[node]])
    return torch.reshape(c, (2, 1))


# initialize s, t arrays
def prepare():
    global s_v, t_v
    s_v = np.zeros(len(initial_graph.nodes) + 1)
    t_v = np.zeros(len(initial_graph.nodes) + 1)
    for ter in terminals:
        t_v[ter] = 1


def compute_large_constant():
    edges = initial_graph.edges
    global const
    const = 0
    for (u, v) in edges:
        const += initial_graph[u][v]['weight']


# checks if all terminals are covered in the current solution
def termination():
    done = True
    for node in terminals:
        # if node not in self.qNetwork_env.S:
        if node not in steiner_tree.nodes():
            done = False
            break
    return done


def postprocessing(st, t):
    done = False
    copy = st.copy()
    while done is not True:
        done = True
        for node in copy.nodes:
            if st.degree[node] == 1 and node not in t:
                # print("Removing node {}".format(node))
                st.remove_node(node)
                done = False
        copy = st.copy()

    while not nx.is_tree(st):
        edges_cycle = list(nx.find_cycle(st))
        for (u, v) in edges_cycle:
            st.remove_edge(u, v)
            break

    weight = 0
    for u, v in st.edges:
        weight += st[u][v]['weight']

    return st, weight


class QNetwork(nn.Module):

    def __init__(self, p, k):
        # initialize parameters and build model.
        super(QNetwork, self).__init__()

        self.p = p

        self.theta1 = nn.Parameter(torch.empty(self.p, 2), requires_grad=True)
        self.theta2 = nn.Parameter(torch.empty(self.p, k), requires_grad=True)
        self.theta3 = nn.Parameter(torch.empty(2 * self.p, 1), requires_grad=True)
        self.theta4 = nn.Parameter(torch.empty(self.p, self.p), requires_grad=True)
        self.theta5 = nn.Parameter(torch.empty(self.p, self.p), requires_grad=True)
        self.h_theta = nn.Parameter(torch.empty(self.p, 2 * self.p), requires_grad=True)

        self.mu = dict()
        self.mu_prime = dict()
        self.Q = dict()

        self.S = []
        self.S_star = set()
        self.initialize_weights()

    # called only by the env network
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.theta1)
        nn.init.xavier_uniform_(self.theta2)
        nn.init.xavier_uniform_(self.theta3)
        nn.init.xavier_uniform_(self.theta4)
        nn.init.xavier_uniform_(self.theta5)
        nn.init.xavier_uniform_(self.h_theta)

    # updates Q values for all nodes
    def forward(self):
        # compute mu_v for each node - shape (p, 1)
        global one_it
        if used_prev_weights is False or one_it is True:
            for node in initial_graph.nodes:
                connection = connection_op(node)
                norm = np.linalg.norm(np.transpose(matrix[node]))
                if norm == 0:
                    norm = 1
                x_v = torch.from_numpy(np.transpose(matrix[node]) / norm)

                input_relu = torch.add(torch.matmul(self.theta1, connection), torch.matmul(self.theta2, x_v.float()))
                mu_v = F.relu(input_relu)
                self.mu[node] = mu_v

            # compute mu_v_prime for each node - shape (p, 1)
            for node in initial_graph.nodes:
                neighbors = neighborhoods[node]
                s = torch.zeros(self.p, 1)
                for neighbor in neighbors:
                    s = torch.add(s, torch.subtract(self.mu[node], self.mu[neighbor]))
                node_info = torch.cat((self.mu[node], s))
                self.mu_prime[node] = torch.matmul(self.h_theta, F.relu(node_info))

            # compute Q values for each node
            sum_mu_prime = torch.zeros(self.p, 1)
            for node in initial_graph.nodes:
                sum_mu_prime = torch.add(sum_mu_prime, self.mu_prime[node])

            for node in initial_graph.nodes:
                up = torch.matmul(self.theta4, sum_mu_prime)
                down = torch.matmul(self.theta5, self.mu_prime[node])
                node_info = torch.cat((up, down))
                relu_res = F.relu(node_info)
                self.Q[node] = torch.matmul(torch.transpose(self.theta3, 0, 1), relu_res)

            # global one_it
            # one_it = True

        else:
            print("here")
            if one_it is False:
                with open('weights.csv', newline='') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    weights = []
                    for row in spamreader:
                        weights.append(row)
                    index = 0
                    for node in initial_graph.nodes:
                        self.Q[node] = Variable(torch.as_tensor([[float(weights[index][0])]]), requires_grad=True)
                        print(self.Q[node])
                        index += 1

                    one_it = True

        return self.Q


class Agent:
    def __init__(self, graph, t, p, k):

        global initial_graph, terminals, uncovered_terminals
        initial_graph = graph
        terminals = t.copy()
        uncovered_terminals = t.copy()

        prepare()
        compute_large_constant()
        compute_neighborhoods()
        compute_similarity_matrix()

        self.qNetwork_env = QNetwork(p, k)
        # self.qNetwork_env.initialize_weights()

        self.qNetwork_target = QNetwork(p, k)
        # copy the weights of qNetwork_local to qNetwork_target
        self.hard_update(self.qNetwork_env, self.qNetwork_target)

        self.current_cost = 0

        self.optimizer = optim.SGD(self.qNetwork_env.parameters(), lr=LR, momentum=MOMENTUM)

        self.memory = ReplayMemory(BUFFER_SIZE, BATCH_SIZE)

        # initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    # add node to current state S, update S_star with new neighbors
    def transition(self, node):
        global s_v
        s_v[node] = 1
        steiner_tree.add_node(node)

        if node in terminals:
            global uncovered_terminals, covered_terminals
            if node in uncovered_terminals:
                uncovered_terminals.remove(node)
                covered_terminals.append(node)

        self.qNetwork_env.S.append(node)
        if node in self.qNetwork_env.S_star:
            self.qNetwork_env.S_star.remove(node)

        # add to S_star the neighbors of the node
        neighbors = neighborhoods[node]
        for n in neighbors:
            parent[n] = node

        self.qNetwork_env.S_star = self.qNetwork_env.S_star.union(set(neighbors))

        # remove node from S_star s.t. we do not choose it again
        if node in self.qNetwork_env.S_star:
            self.qNetwork_env.S_star.remove(node)

    # calculate new cost of the current solution
    def update_cost(self, node):
        if len(self.qNetwork_env.S) > 0:
            min_weight = math.inf
            # choose the edge with the minimum weight that connects the node to the already covered nodes
            for n in self.qNetwork_env.S:
                if n != node:
                    if initial_graph.has_edge(n, node):
                        w = initial_graph[n][node]["weight"]
                        if w < min_weight:
                            min_weight = w
            if min_weight == math.inf:
                min_weight = 0

        # for the first added node
        else:
            min_weight = 0

        self.current_cost += min_weight

    # calculate reward of the action - case distinction based on whether node is terminal
    def compute_reward(self, node, cost):
        norm = np.linalg.norm(np.transpose(matrix[node]))
        if norm == 0:
            norm = 1
        x_v = np.transpose(matrix[node]) / norm
        length_x_v = np.sum(x_v)

        self.update_cost(node)

        if t_v[node] == 1:
            reward = self.current_cost - cost - length_x_v + const
        else:
            reward = self.current_cost - cost - length_x_v
        return reward

    # perform transition, add experience in memory, eventually learn
    def step(self, node):

        self.transition(node)

        # save experience in replay memory
        reward = self.compute_reward(node, self.current_cost)
        if node in self.qNetwork_env.S:
            self.qNetwork_env.S.remove(node)
        state_t = self.qNetwork_env.S.copy()
        self.qNetwork_env.S.append(node)
        state_t_next = self.qNetwork_env.S.copy()
        self.memory.add(state_t, node, reward, state_t_next)

        # learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # if enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(node, reward, experience)

    # returns a node for given state as per current policy (node = argmax Q(node), Q from env qnetwork)
    def act(self, eps=0.1):
        # eps (float): epsilon, for epsilon-greedy action selection

        self.qNetwork_env.eval()
        self.qNetwork_env.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            if len(self.qNetwork_env.S_star) == 0:
                chosen_node = max(filter(lambda e: e not in self.qNetwork_env.S, self.qNetwork_env.Q),
                                  key=self.qNetwork_env.Q.get)
            else:
                chosen_node = max(
                    filter(lambda e: e not in self.qNetwork_env.S and e in self.qNetwork_env.S_star,
                           self.qNetwork_env.Q),
                    key=self.qNetwork_env.Q.get)

        else:  # randomly choose a node from S_star
            if len(self.qNetwork_env.S_star) > 0:
                chosen_node = random.choice(list(self.qNetwork_env.S_star))
            else:
                chosen_node = random.choice(uncovered_terminals)

        # print("Next node: {}".format(chosen_node))

        steiner_tree.add_edge(chosen_node, parent[chosen_node],
                              weight=initial_graph[chosen_node][parent[chosen_node]]['weight'])
        # print("Added edge ({}, {}) with weight {}".format(chosen_node, parent[chosen_node],
        #                                                   initial_graph[chosen_node][parent[chosen_node]]['weight']))

        return chosen_node

    # perform SGD, update weights, copy weights of env network to target network
    def learn(self, node, reward, experience):
        # print("UPDATING PARAMETERS...")
        # update value parameters using given batch of experience tuples.

        current_state, vertex, reward, next_state = experience

        # update distance matrix
        compute_similarity_matrix()
        self.qNetwork_env.forward()

        criterion = torch.nn.MSELoss()
        # local model is one which we need to train, so it is in training mode
        self.qNetwork_env.train()

        # target model is one with which we need to get our target, so it is in evaluation mode,
        # so that when we do a forward pass with target model it does not calculate gradient
        # we will update target model weights with hard_update function
        self.qNetwork_target.eval()

        # compute max Q′(S_{t+1}, v′; θ′) with θ′ from the target network
        self.qNetwork_target.S = next_state
        Q_prime = self.qNetwork_target.forward()
        Q_value = max(filter(lambda e: e not in self.qNetwork_target.S, Q_prime), key=Q_prime.get)

        y = GAMMA * Q_value + reward

        input = self.qNetwork_env.Q[node]

        target = torch.Tensor([[y]])

        loss = criterion(input, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.hard_update(self.qNetwork_env, self.qNetwork_target)

    # update the weights of the target model based on the weights of the local model - 2 possibilities
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        :param local_model: weights will be copied from
        :param target_model: weights will be copied to
        :param tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    # used in the paper - default here
    def hard_update(self, local_model, target_model):
        """ Copy the weights of qNetwork_local to qNetwork_target
        :param local_model: weights will be copied from
        :param target_model: weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def naive_step(self):
        self.qNetwork_env.forward()

        # compute Q'

        if len(self.qNetwork_env.S_star) == 0:
            chosen_node = max(filter(lambda e: e not in self.qNetwork_env.S, self.qNetwork_env.Q),
                              key=self.qNetwork_env.Q.get)
        else:
            chosen_node = max(
                filter(lambda e: e not in self.qNetwork_env.S and e in self.qNetwork_env.S_star, self.qNetwork_env.Q),
                key=self.qNetwork_env.Q.get)

        # print("chosen node: ")
        # print(chosen_node)

        self.transition(chosen_node)

        if len(self.qNetwork_env.S) == 1:  # first node added
            r = self.compute_reward(chosen_node, 0)
        else:
            r = self.compute_reward(chosen_node, self.current_cost)

        if not termination():
            next_node = max(
                filter(lambda e: e not in self.qNetwork_env.S and e in self.qNetwork_env.S_star, self.qNetwork_env.Q),
                key=self.qNetwork_env.Q.get)
            y = GAMMA * self.qNetwork_env.Q[next_node] + r

            loss = nn.MSELoss()
            input = self.qNetwork_env.Q[chosen_node]
            target = y

            self.optimizer.zero_grad()
            output = loss(input, target)
            output.backward()
            self.optimizer.step()


class ReplayMemory:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["current_state",  # S_t
                                                                 "vertex",
                                                                 "reward",
                                                                 "next_state"])  # S_{t+1}

    def add(self, current_state, v, r, next_state):
        e = self.experiences(current_state, v, r, next_state)
        self.memory.append(e)

    # sample one experience from the memory
    def sample(self):
        experience = random.sample(self.memory, k=self.batch_size)

        current_state = experience[0].current_state
        vertex = experience[0].vertex
        reward = experience[0].reward
        next_state = experience[0].next_state

        return current_state, vertex, reward, next_state

    def __len__(self):
        return len(self.memory)


def dqn(g, t, n_episodes):
    p = 2
    k = 2

    agent = Agent(g, t, p, k)

    # choose random terminal to start with
    node = random.choice(t)

    steiner_tree.add_node(node)
    neighbors = neighborhoods[node]
    for n in neighbors:
        parent[n] = node

    agent.qNetwork_env.forward()
    agent.transition(node)

    # add to memory
    reward = agent.compute_reward(node, agent.current_cost)
    if node in agent.qNetwork_env.S:
        agent.qNetwork_env.S.remove(node)
    state_t = agent.qNetwork_env.S.copy()
    agent.qNetwork_env.S.append(node)
    state_t_next = agent.qNetwork_env.S.copy()
    agent.memory.add(state_t, node, reward, state_t_next)

    while not termination():
        next_node = agent.act()
        agent.step(next_node)

    res, weight = postprocessing(steiner_tree, t)

    return res, weight

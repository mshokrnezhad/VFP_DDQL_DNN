import config
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
rnd = np.random
NETWORK_INIT = config.NETWORK_INIT  # Default configs
NETWORK_SAMPLE = config.NETWORK_SAMPLE  # Sample networks


class Network:
    def __init__(self, INPUT):
        # Building INPUT
        self.INPUT = {key: INPUT[key] if key in INPUT else NETWORK_INIT[key] for key in NETWORK_INIT.keys()}
        # Defining base variables
        self.SEED = self.INPUT["SEED"]
        self.NUM_NODES = self.INPUT["NUM_NODES"]
        self.NUM_TIERS = self.INPUT["NUM_TIERS"]
        self.TIER_HEIGHT = self.INPUT["TIER_HEIGHT"]
        self.TIER_WIDTH = self.INPUT["TIER_WIDTH"]
        self.DC_CAPACITY_MU = self.INPUT["DC_CAPACITY_MU"]
        self.DC_CAPACITY_SIGMA = self.INPUT["DC_CAPACITY_SIGMA"]
        self.DC_CAPACITY_GROWTH_RATE = self.INPUT["DC_CAPACITY_GROWTH_RATE"]
        self.DC_COST_MU = self.INPUT["DC_COST_MU"]
        self.DC_COST_SIGMA = self.INPUT["DC_COST_SIGMA"]
        self.DC_COST_DECREASE_RATE = self.INPUT["DC_COST_DECREASE_RATE"]
        self.LINK_BW_MU = self.INPUT["LINK_BW_MU"]
        self.LINK_BW_SIGMA = self.INPUT["LINK_BW_SIGMA"]
        self.LINK_COST_MU = self.INPUT["LINK_COST_MU"]
        self.LINK_COST_SIGMA = self.INPUT["LINK_COST_SIGMA"]
        self.BURST_SIZE_LIMIT = self.INPUT["BURST_SIZE_LIMIT"]
        self.PACKET_SIZE = self.INPUT["PACKET_SIZE"]
        self.NUM_PRIORITY_LEVELS = self.INPUT["NUM_PRIORITY_LEVELS"]
        self.NUM_PATHS_UB = self.INPUT["NUM_PATHS_UB"]
        self.LINK_LENGTH_UB = self.INPUT["LINK_LENGTH_UB"]
        self.SAMPLE = self.INPUT["SAMPLE"]
        # Defining complementary variables
        rnd.seed(self.SEED)
        self.PRIORITIES = np.linspace(0, self.NUM_PRIORITY_LEVELS, self.NUM_PRIORITY_LEVELS + 1).astype(int)
        self.NODES = np.arange(self.NUM_NODES)
        self.NODE_TIERS = np.array([self.get_tier_num(i) for i in self.NODES])
        self.X_LOCS, self.Y_LOCS = self.initialize_coordinates()
        self.DISTANCES = self.find_distances()
        self.DC_CAPACITIES = self.initialize_dc_capacities()
        self.DC_COSTS = self.initialize_dc_costs()
        self.BURST_SIZE_LIMIT_PER_PRIORITY, self.BURST_SIZE_CUM_LIMIT_PER_PRIORITY = self.find_burst_size_limit_per_priority()
        self.NUM_LINKS, self.LINKS_LIST, self.LINKS_MATRIX = self.initialize_links()
        self.LINK_BWS, self.LINK_BWS_MATRIX, self.LINK_BWS_LIMIT_PER_PRIORITY, self.LINK_BWS_CUM_LIMIT_PER_PRIORITY = self.initialize_link_bws()
        self.LINK_COSTS, self.LINK_COSTS_MATRIX = self.initialize_link_costs()
        self.LINK_DELAYS, self.LINK_DELAYS_MATRIX = self.initialize_link_delays()
        self.FIRST_TIER_NODES = self.get_first_tier_nodes()
        self.NUM_PATHS, self.PATHS_LIST = self.find_all_paths()  # PATHS_PER_HEAD[i] denotes paths that begin at node i, PATHS_PER_TAIL[i] denotes paths that end at node i
        self.LINKS_PATHS_MATRIX = self.match_paths_to_links()
        self.MAX_COST_PER_TIER = self.find_max_cost_per_tier()
        self.MIN_COST_PER_TIER = self.find_min_cost_per_tier()

    def initialize_coordinates(self):
        if self.SAMPLE == "":
            X_LOCS = np.array([rnd.randint(self.get_tier_num(i) * self.TIER_WIDTH, (self.get_tier_num(i) + 1) * self.TIER_WIDTH) for i in self.NODES])
            Y_LOCS = np.random.randint(0, self.TIER_HEIGHT, self.NUM_NODES)
        else:
            X_LOCS = NETWORK_SAMPLE[self.SAMPLE]["X_LOCS"]
            Y_LOCS = NETWORK_SAMPLE[self.SAMPLE]["Y_LOCS"]

        return X_LOCS, Y_LOCS

    def initialize_dc_capacities(self):
        if self.SAMPLE == "" or NETWORK_SAMPLE[self.SAMPLE]["DC_CAPACITIES"] == []:
            mu = self.DC_CAPACITY_MU
            sigma = self.DC_CAPACITY_SIGMA
            rate = self.DC_CAPACITY_GROWTH_RATE
            nodes_mu = [mu + (self.NODE_TIERS[i] * rate) for i in self.NODES]

            dc_capacities = np.array([np.random.normal(nodes_mu[i], sigma, 1)[0].round(0).astype(int) for i in self.NODES])
        else:
            dc_capacities = np.array(NETWORK_SAMPLE[self.SAMPLE]["DC_CAPACITIES"])

        return dc_capacities

    def update_dc_capacities(self, node, requirement):
        self.DC_CAPACITIES[node] -= requirement

    def initialize_dc_costs(self):
        mu = self.DC_COST_MU
        sigma = self.DC_COST_SIGMA
        rate = self.DC_COST_DECREASE_RATE
        nodes_mu = [mu - (self.NODE_TIERS[i] * rate) for i in self.NODES]

        dc_costs = np.array([np.random.normal(nodes_mu[i], sigma, 1)[0].round(0).astype(int) for i in self.NODES])

        return dc_costs

    def initialize_links(self):
        links_list = []
        links_matrix = np.zeros((self.NUM_NODES, self.NUM_NODES))

        if self.SAMPLE == "":
            for i in self.NODES:
                for j in self.NODES:
                    if i != j and self.is_j_neighbor_of_i(i, j):
                        if (i, j) not in links_list:
                            links_list.append((i, j))
                        if (j, i) not in links_list:
                            links_list.append((j, i))
        else:
            links_list = NETWORK_SAMPLE[self.SAMPLE]["LINKS_LIST"]

        for (i, j) in links_list:
            links_matrix[i, j] = 1

        num_links = len(links_list)

        return num_links, links_list, links_matrix

    def initialize_link_bws(self):
        link_bws_matrix = np.zeros((self.NUM_NODES, self.NUM_NODES)).astype(int)
        link_bws = np.zeros(self.NUM_LINKS).astype(int)
        link_bws_limit_per_priority = np.zeros((self.NUM_LINKS, self.NUM_PRIORITY_LEVELS + 1)).astype(int)
        link_bws_cum_limit_per_priority = np.zeros((self.NUM_LINKS, self.NUM_PRIORITY_LEVELS + 1)).astype(int)
        mu = self.LINK_BW_MU
        sigma = self.LINK_BW_SIGMA

        for (i, j) in self.LINKS_LIST:
            link_index = self.LINKS_LIST.index((i, j))
            rnd_bw = np.random.normal(mu, sigma, 1)[0].round(0).astype(int)
            # link_bws_dict[(i, j)] = rnd_bw
            # link_bws_dict[(j, i)] = rnd_bw
            link_bws_matrix[i, j] = rnd_bw
            link_bws[link_index] = rnd_bw

        link_bws_matrix = link_bws_matrix.astype(int)

        for (i, j) in self.LINKS_LIST:
            link_index = self.LINKS_LIST.index((i, j))
            for n in self.PRIORITIES:
                if n > 0:
                    limit = int(((self.NUM_PRIORITY_LEVELS + 1 - n) / np.array(self.PRIORITIES).sum()) * link_bws[link_index])
                    link_bws_limit_per_priority[link_index, n] = limit
                else:
                    link_bws_limit_per_priority[link_index, n] = 0

        for (i, j) in self.LINKS_LIST:
            link_index = self.LINKS_LIST.index((i, j))
            array = []
            for n in self.PRIORITIES:
                array.append(link_bws_limit_per_priority[link_index, n])
            for n in self.PRIORITIES:
                link_bws_cum_limit_per_priority[link_index, n] = np.array(array)[:n].sum()
                link_bws_cum_limit_per_priority[link_index, n] = np.array(array)[:n].sum()

        return link_bws, link_bws_matrix, link_bws_limit_per_priority, link_bws_cum_limit_per_priority

    def update_link_bws(self, priority, _path, bw_requirement):  # It updates LINK_BWS, LINK_BWS_MATRIX, LINK_BWS_LIMIT_PER_PRIORITY, and LINK_BWS_CUM_LIMIT_PER_PRIORITY after allocating a priority and a path to a request.
        if(_path != {}):
            path = self.PATHS_LIST.index(_path)
            link_indexes = [i for i in range(self.NUM_LINKS) if self.LINKS_PATHS_MATRIX[i][path] == 1]

            for index in link_indexes:
                (i, j) = self.LINKS_LIST[index]
                self.LINK_BWS[index] -= bw_requirement
                self.LINK_BWS_MATRIX[i, j] -= bw_requirement
                self.LINK_BWS_LIMIT_PER_PRIORITY[index, priority] -= bw_requirement
                
                link_bws_cum_limit_per_priority = np.zeros((self.NUM_LINKS, self.NUM_PRIORITY_LEVELS + 1)).astype(int)
                for (i, j) in self.LINKS_LIST:
                    link_index = self.LINKS_LIST.index((i, j))
                    array = []
                    for n in self.PRIORITIES:
                        array.append(self.LINK_BWS_LIMIT_PER_PRIORITY[link_index, n])
                    for n in self.PRIORITIES:
                        link_bws_cum_limit_per_priority[link_index, n] = np.array(array)[:n].sum()
                        link_bws_cum_limit_per_priority[link_index, n] = np.array(array)[:n].sum()
                self.LINK_BWS_CUM_LIMIT_PER_PRIORITY = link_bws_cum_limit_per_priority

    def initialize_link_costs(self):
        link_costs_matrix = np.zeros((self.NUM_NODES, self.NUM_NODES)).astype(int)
        link_costs = np.zeros(self.NUM_LINKS).astype(int)
        mu = self.LINK_COST_MU
        sigma = self.LINK_COST_SIGMA

        for (i, j) in self.LINKS_LIST:
            link_index = self.LINKS_LIST.index((i, j))
            rnd_cost = np.random.normal(mu, sigma, 1)[0].round(0).astype(int)
            link_costs_matrix[i, j] = rnd_cost
            link_costs[link_index] = rnd_cost

        return link_costs, link_costs_matrix

    def initialize_link_delays(self):  # For more details, check the source paper.
        link_delays_matrix = np.ones((self.NUM_PRIORITY_LEVELS + 1, self.NUM_NODES, self.NUM_NODES)) * 10
        link_delays = np.zeros((self.NUM_LINKS, self.NUM_PRIORITY_LEVELS + 1))

        for (i, j) in self.LINKS_LIST:
            link_index = self.LINKS_LIST.index((i, j))
            for n in self.PRIORITIES:
                if n > 0:
                    delay = ((self.BURST_SIZE_CUM_LIMIT_PER_PRIORITY[n] + self.PACKET_SIZE) / (self.LINK_BWS[link_index] - self.LINK_BWS_CUM_LIMIT_PER_PRIORITY[link_index, n])) + self.PACKET_SIZE / self.LINK_BWS[link_index]
                    link_delays_matrix[n, i, j] = round(delay, 3)
                    link_delays[link_index, n] = round(delay, 3)
                else:
                    link_delays[link_index, n] = 10

        return link_delays, link_delays_matrix

    def find_burst_size_limit_per_priority(self):  # For more details, check the source paper.
        burst_size_limit_per_priority = np.array([((self.NUM_PRIORITY_LEVELS + 1 - i) / np.array(self.PRIORITIES).sum()) * self.BURST_SIZE_LIMIT if i > 0 else 0 for i in self.PRIORITIES]).astype(int)
        # link_bursts = np.array([burst_size_limit_per_priority for l in self.LINKS]).astype(int)
        burst_size_cum_limit_per_priority = np.array([np.array(burst_size_limit_per_priority[:i + 1]).sum() for i in self.PRIORITIES]).astype(int)

        return burst_size_limit_per_priority, burst_size_cum_limit_per_priority

    def update_burst_size_limit_per_priority(self, priority, burst_size):  # It updates BURST_SIZE_LIMIT_PER_PRIORITY after allocating a priority to a request.
        self.BURST_SIZE_LIMIT_PER_PRIORITY[priority] -= burst_size

    def get_state(self):
        node_features = []

        for i in self.NODES:
            node_feature = []
            node_feature.append(self.DC_CAPACITIES[i])
            node_feature.append(self.DC_COSTS[i])
            node_features.append(node_feature)

        link_features = {}
        for (i, j) in self.LINKS_LIST:
            link_index = self.LINKS_LIST.index((i, j))
            link_features[(i, j)] = []
            link_features[(i, j)].append(self.LINK_BWS[link_index])
            link_features[(i, j)].append(self.LINK_COSTS[link_index])
            for k in self.PRIORITIES:
                if k not in [0]:
                    link_features[(i, j)].append(self.LINK_DELAYS[link_index][k])

        return node_features, self.LINKS_MATRIX, link_features

    def update_state(self, action={}):  # It updates the network state after receiving an action from a request.
        node = action["node"] if "node" in action.keys() else ""
        dc_capacity_requirement = action["dc_capacity_requirement"] if "dc_capacity_requirement" in action.keys() else ""
        priority = action["priority"] if "priority" in action.keys() else ""
        burst_size = action["burst_size"] if "burst_size" in action.keys() else ""
        bw_requirement = action["bw_requirement"] if "bw_requirement" in action.keys() else ""
        req_path = action["req_path"] if "req_path" in action.keys() else ""
        rpl_path = action["rpl_path"] if "rpl_path" in action.keys() else ""

        if node != "":
            self.update_dc_capacities(node, dc_capacity_requirement)
        if priority != "":
            if burst_size != "":
                self.update_burst_size_limit_per_priority(priority, burst_size)
            if req_path != "":
                self.update_link_bws(priority, req_path, bw_requirement)
            if rpl_path != "":
                self.update_link_bws(priority, rpl_path, bw_requirement)

    def get_tier_num(self, i):
        tier_num = 0
        tier_size = math.ceil(self.NUM_NODES / self.NUM_TIERS)

        for t in range(self.NUM_TIERS):
            if t * tier_size <= i <= (t + 1) * tier_size:
                tier_num = t

        return tier_num

    def find_distances(self):
        distances = np.array([[np.hypot(self.X_LOCS[i] - self.X_LOCS[j], self.Y_LOCS[i] - self.Y_LOCS[j]) for j in self.NODES] for i in self.NODES])
        distances = distances.astype(int)

        return distances

    def plot(self):
        G = nx.Graph()
        G.add_edges_from(self.LINKS_LIST)
        pos = {i: (self.X_LOCS[i], self.Y_LOCS[i]) for i in self.NODES}

        nx.draw_networkx(G, pos=pos)
        plt.show()

    def is_j_neighbor_of_i(self, i, j):
        if abs(self.NODE_TIERS[i] - self.NODE_TIERS[j]) <= 1:
            close_neighbors = {k: self.DISTANCES[i, k] for k in self.NODES if self.NODE_TIERS[k] == self.NODE_TIERS[j] and k != i}
            if j == min(close_neighbors, key=close_neighbors.get):
                return True
            else:
                return False
        else:
            return False

    def is_connected(self):
        connected = np.zeros(self.NUM_NODES).astype(int)
        visited = np.zeros(self.NUM_NODES).astype(int)
        connected[0] = 1

        for k in self.NODES:
            for i in self.NODES:
                if connected[i] == 1 and visited[i] == 0:
                    visited[i] = 1
                    for j in self.NODES:
                        if self.LINKS_MATRIX[i][j] == 1:
                            connected[j] = 1

        if np.sum(connected) == self.NUM_NODES:
            return True

    def get_first_tier_nodes(self):
        return np.array([i for i in self.NODES if self.get_tier_num(i) == 0])

    def find_all_paths_per_node_pair(self, start, end, count, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start >= self.NUM_NODES or end >= self.NUM_NODES:
            return []
        paths = []
        for i in self.NODES:
            if self.LINKS_MATRIX[start][i] == 1 and i not in path:
                if count < self.LINK_LENGTH_UB:
                    # count += 1
                    new_paths = self.find_all_paths_per_node_pair(i, end, count + 1, path)
                    for new_path in new_paths:
                        paths.append(new_path)
        return paths

    def find_all_paths(self):
        paths_list = []

        for i in self.NODES:
            for j in self.NODES:
                if j != i:
                    new_paths = self.find_all_paths_per_node_pair(i, j, 0)
                    path_costs = np.zeros(len(new_paths)).astype(int)

                    for p in range(len(new_paths)):
                        cost = 0
                        for v in range(len(new_paths[p]) - 1):
                            link_index = self.LINKS_LIST.index((new_paths[p][v], new_paths[p][v + 1]))
                            cost += self.LINK_COSTS[link_index]
                        path_costs[p] = cost

                    for p in range(self.NUM_PATHS_UB):
                        if(len(path_costs) > 0):
                            min_index = path_costs.argmin()
                            paths_list.append(new_paths[min_index])
                            path_costs[min_index] = 1000000

        num_paths = len(paths_list)

        return num_paths, paths_list

    def match_paths_to_links(self):
        links_paths_matrix = np.zeros((self.NUM_LINKS, self.NUM_PATHS)).astype(int)

        for link_index in range(self.NUM_LINKS):
            for path_index in range(self.NUM_PATHS):
                for node_index in range(len(self.PATHS_LIST[path_index]) - 1):
                    if self.LINKS_LIST[link_index][0] == self.PATHS_LIST[path_index][node_index] and self.LINKS_LIST[link_index][1] == self.PATHS_LIST[path_index][node_index + 1]:
                        links_paths_matrix[link_index][path_index] = 1

        return links_paths_matrix
    
    def find_max_cost_per_tier(self):
        entry_nodes = self.get_first_tier_nodes()
        costs = {}
        for e in entry_nodes:
            tiers_cost = {}
            for t in range(self.NUM_TIERS):
                nodes_cost = {}
                for v in self.NODES:
                    if self.get_tier_num(v) == t:
                        c1 = 0
                        for qp in self.PATHS_LIST:
                            if qp[0] == e and qp[-1] == v:
                                c2 = 0
                                for l in np.where(self.LINKS_PATHS_MATRIX[:, qp] == 1)[0]:
                                    c2 += self.LINK_COSTS[l]
                                for pp in self.PATHS_LIST:
                                    if pp[0] == v and pp[-1] == e:
                                        c3 = 0
                                        for l in np.where(self.LINKS_PATHS_MATRIX[:, pp] == 1)[0]:
                                            c3 += self.LINK_COSTS[l]
                                        if c2 + c3 > c1:
                                            c1 = c2 + c3
                        c1 += self.DC_COSTS[v]
                        nodes_cost[v] = c1
                tiers_cost[t] = max(nodes_cost.values())
            costs[e] = tiers_cost
        return costs

    def find_min_cost_per_tier(self):
        entry_nodes = self.get_first_tier_nodes()
        costs = {}
        for e in entry_nodes:
            tiers_cost = {}
            for t in range(self.NUM_TIERS):
                nodes_cost = {}
                for v in self.NODES:
                    if self.get_tier_num(v) == t:
                        c1 = 10e10
                        for qp in self.PATHS_LIST:
                            if qp[0] == e and qp[-1] == v:
                                c2 = 0
                                for l in np.where(self.LINKS_PATHS_MATRIX[:, qp] == 1)[0]:
                                    c2 += self.LINK_COSTS[l]
                                for pp in self.PATHS_LIST:
                                    if pp[0] == v and pp[-1] == e:
                                        c3 = 0
                                        for l in np.where(self.LINKS_PATHS_MATRIX[:, pp] == 1)[0]:
                                            c3 += self.LINK_COSTS[l]
                                        if c2 + c3 < c1:
                                            c1 = c2 + c3
                        c1 = self.DC_COSTS[v] if c1 == 10e10 else c1 + self.DC_COSTS[v]
                        nodes_cost[v] = c1
                tiers_cost[t] = min(nodes_cost.values())
            costs[e] = tiers_cost
        return costs

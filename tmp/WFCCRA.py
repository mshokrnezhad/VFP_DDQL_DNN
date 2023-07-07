import numpy as np
import copy


class WFCCRA:

    def __init__(self, net_obj, req_obj, srv_obj, REQUESTED_SERVICES, REQUESTS_ENTRY_NODES):
        self.net_obj = net_obj
        self.req_obj = req_obj
        self.srv_obj = srv_obj
        self.REQUESTED_SERVICES = REQUESTED_SERVICES
        self.REQUESTS_ENTRY_NODES = REQUESTS_ENTRY_NODES
        self.EPSILON = 0.001

    def sort_requests(self):
        # sort request indexes in terms of delay requirements
        sorted_requirements = np.sort(self.req_obj.DELAY_REQUIREMENTS)
        sorted_requests = []

        for requirement in sorted_requirements:
            sorted_requests.append(np.where(self.req_obj.DELAY_REQUIREMENTS == requirement)[0][0])

        return sorted_requests

    def solve_per_req(self, action, switch):
        r = action['req_id']
        v = action['node_id']

        g = np.zeros(self.net_obj.NUM_NODES)
        rho = np.zeros(len(self.net_obj.PRIORITIES))
        req_path = np.zeros((len(self.net_obj.PATHS), len(self.net_obj.PRIORITIES)))
        rpl_path = np.zeros((len(self.net_obj.PATHS), len(self.net_obj.PRIORITIES)))

        resources_per_req = []
        costs_per_req = []
        delays_per_req = []
        # cost_details_per_req = []

        if self.req_obj.CAPACITY_REQUIREMENTS[r] <= self.net_obj.DC_CAPACITIES[v]:
            g[v] = 1
            for k in self.net_obj.PRIORITIES:
                rho[k] = 1
                for p1 in np.intersect1d(self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]], self.net_obj.PATHS_PER_TAIL[v]):
                    flag1 = True
                    for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                        if self.req_obj.BW_REQUIREMENTS[r] > self.net_obj.LINK_BWS[l1]:
                            flag1 = False
                        if self.req_obj.BURST_SIZES[r] > self.net_obj.LINK_BURSTS[l1, k]:
                            flag1 = False
                    if flag1:
                        req_path[p1][k] = 1
                        for p2 in np.intersect1d(self.net_obj.PATHS_PER_HEAD[v], self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]]):
                            flag2 = True
                            for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                                if self.req_obj.BW_REQUIREMENTS[r] > self.net_obj.LINK_BWS[l2] or self.req_obj.BURST_SIZES[r] > self.net_obj.LINK_BURSTS[l2][k]:
                                    flag2 = False
                            if flag2:
                                rpl_path[p2][k] = 1
                                d = 0
                                c = 0
                                # req_paths_cost = 0
                                # rpl_paths_cost = 0
                                for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                                    d += self.net_obj.LINK_DELAYS[l1][k]
                                for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                                    d += self.net_obj.LINK_DELAYS[l2][k]
                                d += self.net_obj.PACKET_SIZE / self.req_obj.CAPACITY_REQUIREMENTS[r]  # d += self.net_obj.PACKET_SIZE / (self.net_obj.DC_CAPACITIES[v] + self.EPSILON)
                                if self.net_obj.get_tier_num(v) == 0 or d <= self.req_obj.DELAY_REQUIREMENTS[r]:
                                    c += self.net_obj.DC_COSTS[v]  # * self.req_obj.CAPACITY_REQUIREMENTS[r]
                                    for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                                        c += self.net_obj.LINK_COSTS[l1]  # * self.req_obj.BW_REQUIREMENTS[r]
                                        # req_paths_cost += self.net_obj.LINK_COSTS[l1]  # * self.req_obj.BW_REQUIREMENTS[r]
                                    for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                                        c += self.net_obj.LINK_COSTS[l2]  # * self.req_obj.BW_REQUIREMENTS[r]
                                        # rpl_paths_cost += self.net_obj.LINK_COSTS[l1]  # * self.req_obj.BW_REQUIREMENTS[r]
                                    resources_per_req.append([v, k, p1, p2])
                                    costs_per_req.append(c)
                                    delays_per_req.append(d)
                                    # cost_details_per_req.append([req_paths_cost, rpl_paths_cost])

        if len(costs_per_req) > 0:
            min_index = np.array(costs_per_req).argmin()
            optimal_resources_per_req = resources_per_req[min_index]
            optimal_cost_per_req = costs_per_req[min_index]
            optimal_delay_per_req = delays_per_req[min_index]

            solution = {
                "pair": (self.REQUESTS_ENTRY_NODES[r], action['node_id']),
                # "g": optimal_resources_per_req[0],
                "priority": optimal_resources_per_req[1],
                "req_path": optimal_resources_per_req[2],
                "rpl_path": optimal_resources_per_req[3],
                "req_path_details": self.net_obj.PATHS_DETAILS[optimal_resources_per_req[2]],
                "rpl_path_details": self.net_obj.PATHS_DETAILS[optimal_resources_per_req[3]],
                "info": "Feasible",
                "OF": optimal_cost_per_req,
                "delay": optimal_delay_per_req,
                "done": False
            }
        elif self.REQUESTS_ENTRY_NODES[r] == v and self.req_obj.CAPACITY_REQUIREMENTS[r] <= self.net_obj.DC_CAPACITIES[v]:
            solution = {
                "pair": (self.REQUESTS_ENTRY_NODES[r], action['node_id']),
                # "g": optimal_resources_per_req[0],
                "priority": {},
                "req_path": {},
                "rpl_path": {},
                "req_path_details": {},
                "rpl_path_details": {},
                "info": "Feasible",
                "OF": self.net_obj.DC_COSTS[v],
                "delay": self.net_obj.PACKET_SIZE / self.req_obj.CAPACITY_REQUIREMENTS[r],
                "done": False
            }
        else:
            solution = {
                "pair": (self.REQUESTS_ENTRY_NODES[r], v),
                # "g": {},
                "priority": {},
                "req_path": {},
                "rpl_path": {},
                "req_path_details": {},
                "rpl_path_details": {},
                "info": "Infeasible",
                "OF": 0,
                "delay": 0,
                "done": True
            }

        """
        DC_CAPACITIES[resources[r][0]] = DC_CAPACITIES[resources[r][0]] - self.req_obj.CAPACITY_REQUIREMENTS[r]
        for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, resources[r][2]] == 1)[0]:
            LINK_BWS[l1] = LINK_BWS[l1] - self.req_obj.BW_REQUIREMENTS[r]
            LINK_BURSTS[l1, resources[r][1]] = LINK_BURSTS[l1, resources[r][1]] - self.req_obj.BURST_SIZES[r]
        for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, resources[r][3]] == 1)[0]:
            LINK_BWS[l2] = LINK_BWS[l2] - self.req_obj.BW_REQUIREMENTS[r]
            LINK_BURSTS[l2, resources[r][1]] = LINK_BURSTS[l2, resources[r][1]] - self.req_obj.BURST_SIZES[r]
        cost_details[r] = cost_details_per_req[min_index]
        """

        return solution

    def solve(self):
        z = np.zeros((self.srv_obj.NUM_SERVICES, self.net_obj.NUM_NODES))
        g = np.zeros((self.req_obj.NUM_REQUESTS, self.net_obj.NUM_NODES))
        rho = np.zeros((self.req_obj.NUM_REQUESTS, len(self.net_obj.PRIORITIES)))
        req_path = np.zeros((self.req_obj.NUM_REQUESTS, len(self.net_obj.PATHS), len(self.net_obj.PRIORITIES)))
        rpl_path = np.zeros((self.req_obj.NUM_REQUESTS, len(self.net_obj.PATHS), len(self.net_obj.PRIORITIES)))

        BASE_DC_CAPACITIES = copy.deepcopy(self.net_obj.DC_CAPACITIES)
        DC_CAPACITIES = self.net_obj.DC_CAPACITIES
        LINK_BWS = self.net_obj.LINK_BWS
        LINK_BURSTS = np.array([self.net_obj.BURST_SIZE_LIMIT_PER_PRIORITY for l in self.net_obj.LINKS])
        # sorted_requests = self.sort_requests()
        sorted_requests = np.argsort(self.req_obj.DELAY_REQUIREMENTS)
        resources = {}
        costs = {}
        delays = {}
        # cost_details = {}

        for r in sorted_requests:
            resources_per_req = []
            costs_per_req = []
            delays_per_req = []
            # cost_details_per_req = []
            for v in self.net_obj.NODES:
                if self.req_obj.CAPACITY_REQUIREMENTS[r] <= DC_CAPACITIES[v]:
                    z[self.REQUESTED_SERVICES[r]][v] = 1
                    g[r][v] = 1
                if z[self.REQUESTED_SERVICES[r]][v] == 1 and g[r][v] == 1:
                    for k in self.net_obj.PRIORITIES:
                        rho[r][k] = 1
                        for p1 in np.intersect1d(self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]], self.net_obj.PATHS_PER_TAIL[v]):
                            flag1 = True
                            for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                                if self.req_obj.BW_REQUIREMENTS[r] > LINK_BWS[l1]:
                                    flag1 = False
                                if self.req_obj.BURST_SIZES[r] > LINK_BURSTS[l1, k]:
                                    flag1 = False
                            if flag1:
                                req_path[r][p1][k] = 1
                                for p2 in np.intersect1d(self.net_obj.PATHS_PER_HEAD[v], self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]]):
                                    flag2 = True
                                    for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                                        if self.req_obj.BW_REQUIREMENTS[r] > LINK_BWS[l2] or self.req_obj.BURST_SIZES[r] > LINK_BURSTS[l2][k]:
                                            flag2 = False
                                    if flag2:
                                        rpl_path[r][p2][k] = 1
                                        d = 0
                                        c = 0
                                        # req_paths_cost = 0
                                        # rpl_paths_cost = 0
                                        for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                                            d += self.net_obj.LINK_DELAYS[l1][k]
                                        for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                                            d += self.net_obj.LINK_DELAYS[l2][k]
                                        d += self.net_obj.PACKET_SIZE / self.req_obj.CAPACITY_REQUIREMENTS[r]  # d += self.net_obj.PACKET_SIZE / (self.net_obj.DC_CAPACITIES[v] + self.EPSILON)
                                        if self.net_obj.get_tier_num(v) == 0 or d <= self.req_obj.DELAY_REQUIREMENTS[r]:
                                            c += self.net_obj.DC_COSTS[v]  # * self.req_obj.CAPACITY_REQUIREMENTS[r]
                                            for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                                                c += self.net_obj.LINK_COSTS[l1]  # * self.req_obj.BW_REQUIREMENTS[r]
                                                # req_paths_cost += self.net_obj.LINK_COSTS[l1]
                                            for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                                                c += self.net_obj.LINK_COSTS[l2]  # * self.req_obj.BW_REQUIREMENTS[r]
                                                # rpl_paths_cost += self.net_obj.LINK_COSTS[l1]
                                            resources_per_req.append([v, k, p1, p2])
                                            costs_per_req.append(c)
                                            delays_per_req.append(d)
                                            # cost_details_per_req.append([req_paths_cost, rpl_paths_cost])
            if len(costs_per_req) > 0:
                min_index = np.array(costs_per_req).argmin()
                resources[r] = resources_per_req[min_index]
                costs[r] = costs_per_req[min_index]
                delays[r] = delays_per_req[min_index]

                DC_CAPACITIES[resources[r][0]] = DC_CAPACITIES[resources[r][0]] - self.req_obj.CAPACITY_REQUIREMENTS[r]
                for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, resources[r][2]] == 1)[0]:
                    LINK_BWS[l1] = LINK_BWS[l1] - self.req_obj.BW_REQUIREMENTS[r]
                    LINK_BURSTS[l1, resources[r][1]] = LINK_BURSTS[l1, resources[r][1]] - self.req_obj.BURST_SIZES[r]
                for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, resources[r][3]] == 1)[0]:
                    LINK_BWS[l2] = LINK_BWS[l2] - self.req_obj.BW_REQUIREMENTS[r]
                    LINK_BURSTS[l2, resources[r][1]] = LINK_BURSTS[l2, resources[r][1]] - self.req_obj.BURST_SIZES[r]
                # cost_details[r] = cost_details_per_req[min_index]
            else:
                resources[r] = [-1, -1, -1, -1]
                costs[r] = 0
                delays[r] = 0

        solution = {"pairs": {}, "priorities": {}, "req_paths": {}, "rpl_paths": {}, "avg_of": 0, "avg_dly": 0, "reqs": 0, "dc_var": 0}
        pairs = {}
        priorities = {}
        req_paths = {}
        rpl_paths = {}
        reqs = 0

        sum_ofs = sum(costs.values())
        sum_delays = sum(delays.values())
        for r in self.req_obj.REQUESTS:
            if costs[r] != 0:
                pairs[r] = (self.REQUESTS_ENTRY_NODES[r], resources[r][0])
                priorities[r] = resources[r][1]
                req_paths[r] = self.net_obj.PATHS_DETAILS[resources[r][2]]
                rpl_paths[r] = self.net_obj.PATHS_DETAILS[resources[r][3]]
            else:
                pairs[r] = -1
                priorities[r] = -1
                req_paths[r] = -1
                rpl_paths[r] = -1

        for key in costs:
            if costs[key] != 0:
                reqs += 1

        solution["pairs"] = pairs
        solution["priorities"] = priorities
        solution["req_paths"] = req_paths
        solution["rpl_paths"] = rpl_paths
        solution["avg_of"] = 0 if reqs == 0 else sum_ofs/reqs
        solution["avg_dly"] = 0 if reqs == 0 else sum_delays/reqs
        solution["reqs"] = reqs
        solution["dc_var"] = np.var(100*(DC_CAPACITIES/BASE_DC_CAPACITIES))

        return solution

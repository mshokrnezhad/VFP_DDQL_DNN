import numpy as np
import copy
import random


class P1BSA:
    def __init__(self, net_obj, req_obj, srv_obj, REQUESTED_SERVICES, REQUESTS_ENTRY_NODES):
        self.net_obj = net_obj
        self.req_obj = req_obj
        self.srv_obj = srv_obj
        self.REQUESTED_SERVICES = REQUESTED_SERVICES
        self.REQUESTS_ENTRY_NODES = REQUESTS_ENTRY_NODES
        self.EPSILON = 0.001

    def sort_requests(self):
        # sort request indexes in terms of delay requirements

        capacity_difference = self.req_obj.CAPACITY_REQUIREMENT_UB - \
            self.req_obj.CAPACITY_REQUIREMENT_LB
        normalized_requirements = [self.req_obj.CAPACITY_REQUIREMENTS[r] + self.req_obj.CAPACITY_REQUIREMENT_LB + (capacity_difference*(
            (self.req_obj.CAPACITY_REQUIREMENTS[r]-self.req_obj.CAPACITY_REQUIREMENT_LB)/(capacity_difference))) for r in self.req_obj.REQUESTS]
        normalized_requirements = np.array(normalized_requirements)

        sorted_requests = normalized_requirements.argsort()[::-1]

        return sorted_requests

    def generate_random_priority(self):
        k = 0
        while k == 0:
            k = random.choice(self.net_obj.PRIORITIES)
            
        return k
        
    
    def solve(self):
        z = np.zeros((self.srv_obj.NUM_SERVICES, self.net_obj.NUM_NODES))
        g = np.zeros((self.req_obj.NUM_REQUESTS, self.net_obj.NUM_NODES))
        rho = np.zeros((self.req_obj.NUM_REQUESTS,
                       len(self.net_obj.PRIORITIES)))
        req_path = np.zeros((self.req_obj.NUM_REQUESTS, len(
            self.net_obj.PATHS), len(self.net_obj.PRIORITIES)))
        rpl_path = np.zeros((self.req_obj.NUM_REQUESTS, len(
            self.net_obj.PATHS), len(self.net_obj.PRIORITIES)))

        BASE_DC_CAPACITIES = copy.deepcopy(self.net_obj.DC_CAPACITIES)
        DC_CAPACITIES = self.net_obj.DC_CAPACITIES
        LINK_BWS = self.net_obj.LINK_BWS
        LINK_BURSTS = np.array(
            [self.net_obj.BURST_SIZE_LIMIT_PER_PRIORITY for l in self.net_obj.LINKS])
        sorted_requests = self.sort_requests()
        # sorted_requests = np.argsort(self.req_obj.DELAY_REQUIREMENTS)
        resources = {}
        costs = {}
        delays = {}
        # cost_details = {}

        for r in sorted_requests:
            resources_per_req = []
            costs_per_req = []
            delays_per_req = []
            # cost_details_per_req = []
            
            flag_node = False
            best_ratio = 0
            for v in self.net_obj.NODES:
                if self.req_obj.CAPACITY_REQUIREMENTS[r] <= DC_CAPACITIES[v]:
                    cur_ratio = self.req_obj.CAPACITY_REQUIREMENTS[r]/DC_CAPACITIES[v]
                    if best_ratio < cur_ratio:
                        flag_node = True
                        z[self.REQUESTED_SERVICES[r]][v] = 1
                        g[r][v] = 1  
                
            if flag_node:
                k = self.generate_random_priority()
                rho[r][k] = 1
                
                req_paths = np.intersect1d(self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]], self.net_obj.PATHS_PER_TAIL[v])
                if(len(req_paths) > 0):
                    p1 = random.choice(req_paths)
                    flag_p1 = True
                    for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                        if self.req_obj.BW_REQUIREMENTS[r] > LINK_BWS[l1] or self.req_obj.BURST_SIZES[r] > LINK_BURSTS[l1, k]:
                            flag_p1 = False
                    if flag_p1:
                        req_path[r][p1][k] = 1
                else:
                    flag_p1 = False  
                    
                rpl_paths = np.intersect1d(self.net_obj.PATHS_PER_HEAD[v], self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]])
                if(len(rpl_paths) > 0):
                    p2 = random.choice(rpl_paths)
                    flag_p2 = True
                    for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                        if self.req_obj.BW_REQUIREMENTS[r] > LINK_BWS[l2] or self.req_obj.BURST_SIZES[r] > LINK_BURSTS[l2][k]:
                            flag_p2 = False
                    if flag_p2:
                        rpl_path[r][p2][k] = 1
                else:
                    flag_p2 = False
                
                if(flag_p1 and flag_p2):
                    d = 0
                    c = 0
                    for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                        d += self.net_obj.LINK_DELAYS[l1][k]
                    for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                        d += self.net_obj.LINK_DELAYS[l2][k]
                    d += self.net_obj.PACKET_SIZE / self.req_obj.CAPACITY_REQUIREMENTS[r]
                    
                    c += self.net_obj.DC_COSTS[v]
                    for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                        c += self.net_obj.LINK_COSTS[l1]
                    for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                        c += self.net_obj.LINK_COSTS[l2]
                        
                    if self.net_obj.get_tier_num(v) == 0 or d <= self.req_obj.DELAY_REQUIREMENTS[r]:    
                        resources_per_req.append([v, k, p1, p2])
                        costs_per_req.append(c)
                        delays_per_req.append(d)  
    
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

from Network import Network
from Request import Request
from Service import Service
import numpy as np
from Functions import parse_state



class Environment:
    def __init__(self, INPUT):
        self.INPUT = INPUT
        self.NET_INPUT = self.INPUT["NET"]
        self.NET_OBJ = Network(self.NET_INPUT)
        self.SRV_INPUT = self.INPUT["SRV"]
        self.SRV_OBJ = Service(self.SRV_INPUT)
        self.REQ_INPUT = self.INPUT["REQ"]
        self.REQ_INPUT["NET_OBJ"] = self.NET_OBJ
        self.REQ_INPUT["SRV_OBJ"] = self.SRV_OBJ
        self.REQ_OBJ = Request(self.REQ_INPUT)

    def get_state(self):
        node_features, links_matrix, link_features = self.NET_OBJ.get_state()
        request_features = self.REQ_OBJ.get_state()

        env_state = {
            "NODE_FEATURES": node_features,
            "LINK_MATRIX": links_matrix,
            "LINK_FEATURES": link_features,
            "REQUEST_FEATURES": request_features
        }

        return env_state

    def update_state(self, action):
        self.NET_OBJ.update_state(action)

    def reset(self, SEED):
        self.NET_INPUT = self.INPUT["NET"]
        self.NET_OBJ = Network(self.NET_INPUT)
        self.SRV_INPUT = self.INPUT["SRV"]
        self.SRV_OBJ = Service(self.SRV_INPUT)
        self.REQ_INPUT = self.INPUT["REQ"]
        self.REQ_INPUT["SEED"] = SEED
        self.REQ_INPUT["NET_OBJ"] = self.NET_OBJ
        self.REQ_INPUT["SRV_OBJ"] = self.SRV_OBJ
        self.REQ_OBJ = Request(self.REQ_INPUT)

    def step(self, action, switch="none"):
        result = self.solve_per_request(action)

        if result["done"]:
            reward = 0
        else:
            max_cost = max(self.NET_OBJ.MAX_COST_PER_TIER[result["pair"][0]].values())
            min_cost = min(self.NET_OBJ.MAX_COST_PER_TIER[result["pair"][0]].values())
            tier_cost_range = max_cost - min_cost
            action_efficiency_range = 100
            action_efficiency = action_efficiency_range - (action_efficiency_range * (result["OF"] - min_cost) / tier_cost_range)
            reward_base = 100
            reward = reward_base ** (self.NET_OBJ.get_tier_num(result["pair"][1]) + 1) + action_efficiency
            _action = {
                "node": result["pair"][1],
                "priority": result["priority"],
                "req_path": result["req_path"],
                "dc_capacity_requirement": self.REQ_OBJ.DC_CAPACITY_REQUIREMENTS[action['req_id']],
                "bw_requirement": self.REQ_OBJ.BW_REQUIREMENTS[action['req_id']]
            }
            self.update_state(_action)

        parsed_resulted_state = parse_state(self.NET_OBJ.NUM_NODES, self.get_state(), self.NET_OBJ)


        return parsed_resulted_state, round(reward, 3), result["done"], result["info"], result["OF"], result["delay"]

    def solve_per_request(self, action):
        entry_node = 0
        request = action['req_id']
        node = action['node_id']

        _solution = {
            "nodes": np.zeros(self.NET_OBJ.NUM_NODES),
            "priorities": np.zeros(len(self.NET_OBJ.PRIORITIES)),
            "request_paths": np.zeros((len(self.NET_OBJ.PATHS_LIST), len(self.NET_OBJ.PRIORITIES))),
            # "reply_paths": np.zeros((len(self.net_obj.PATHS), len(self.net_obj.PRIORITIES)))
        }

        _resources = []
        _costs = []
        _delays = []

        if self.REQ_OBJ.DC_CAPACITY_REQUIREMENTS[request] <= self.NET_OBJ.DC_CAPACITIES[node]:
            _solution["nodes"][node] = 1
            for k in range(1, self.NET_OBJ.NUM_PRIORITY_LEVELS+1):
                _solution["priorities"][k] = 1

                for qp in self.NET_OBJ.PATHS_LIST:
                    if qp[0] == entry_node and qp[-1] == node:
                        for l in np.where(self.NET_OBJ.LINKS_PATHS_MATRIX[:, qp] == 1)[0]:
                            if self.REQ_OBJ.BW_REQUIREMENTS[request] > self.NET_OBJ.LINK_BWS[l]:
                                break
                        else:
                            _solution["request_paths"][qp][k] = 1

                            delay = 0
                            cost = 0

                            for l in np.where(self.NET_OBJ.LINKS_PATHS_MATRIX[:, qp] == 1)[0]:
                                delay += self.NET_OBJ.LINK_DELAYS[l][k]
                            delay += self.NET_OBJ.PACKET_SIZE / self.REQ_OBJ.DC_CAPACITY_REQUIREMENTS[request]

                            if self.NET_OBJ.get_tier_num(node) == 0 or delay <= self.REQ_OBJ.DELAY_REQUIREMENTS[request]:
                                cost += self.NET_OBJ.DC_COSTS[node]
                                for l in np.where(self.NET_OBJ.LINKS_PATHS_MATRIX[:, qp] == 1)[0]:
                                    cost += self.NET_OBJ.LINK_COSTS[l]

                                _resources.append([node, k, qp])
                                _costs.append(cost)
                                _delays.append(delay)

        if len(_costs) > 0:
            min_index = np.array(_costs).argmin()
            resource = _resources[min_index]
            cost = _costs[min_index]
            delay = _delays[min_index]

            solution = {
                "pair": (self.REQ_OBJ.ENTRY_NODES[request], node),
                "priority": resource[1],
                "req_path": resource[2],
                # "rpl_path": resource[3],
                "req_path_details": resource[2],
                # "rpl_path_details": self.NET_OBJ.PATHS_DETAILS[resource[3]],
                "info": "Feasible",
                "OF": cost,
                "delay": delay,
                "done": False
            }
        
        elif self.REQ_OBJ.ENTRY_NODES[request] == node and self.REQ_OBJ.DC_CAPACITY_REQUIREMENTS[request] <= self.NET_OBJ.DC_CAPACITIES[node]:
            solution = {
                "pair": (self.REQ_OBJ.ENTRY_NODES[request], node),
                "priority": {},
                "req_path": {},
                # "rpl_path": {},
                "req_path_details": {},
                # "rpl_path_details": {},
                "info": "Feasible",
                "OF": self.NET_OBJ.DC_COSTS[node],
                "delay": self.NET_OBJ.PACKET_SIZE / self.REQ_OBJ.DC_CAPACITY_REQUIREMENTS[request],
                "done": False
            }
            
        else:
            solution = {
                "pair": (self.REQ_OBJ.ENTRY_NODES[request], node),
                "priority": {},
                "req_path": {},
                # "rpl_path": {},
                "req_path_details": {},
                # "rpl_path_details": {},
                "info": "Infeasible",
                "OF": 0,
                "delay": 0,
                "done": True
            }

        return solution

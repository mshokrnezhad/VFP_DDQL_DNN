from Network import Network
from Request import Request
from ReNet.Service import Service
from Functions import specify_requests_entry_nodes, assign_requests_to_services
from WFCCRA import WFCCRA
import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))


class Environment:
    def __init__(self, NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS):
        self.NUM_NODES = NUM_NODES
        self.NUM_REQUESTS = NUM_REQUESTS
        self.NUM_SERVICES = NUM_SERVICES
        self.NUM_PRIORITY_LEVELS = NUM_PRIORITY_LEVELS
        # self.SWITCH = SWITCH
        self.net_obj = Network(NUM_NODES=NUM_NODES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS)
        # self.REQUESTS_ENTRY_NODES = specify_requests_entry_nodes(self.net_obj.FIRST_TIER_NODES, np.arange(NUM_REQUESTS))
        self.REQUESTS_ENTRY_NODES = np.zeros(self.NUM_REQUESTS).astype("int")
        self.req_obj = Request(NUM_REQUESTS=NUM_REQUESTS, NODES=self.net_obj.NODES, REQUESTS_ENTRY_NODES=self.REQUESTS_ENTRY_NODES)
        self.srv_obj = Service(NUM_SERVICES=NUM_SERVICES)
        self.REQUESTED_SERVICES = assign_requests_to_services(np.arange(NUM_SERVICES), np.arange(NUM_REQUESTS))
        # self.model_obj = CPLEX(self.net_obj, self.req_obj, self.srv_obj, self.REQUESTED_SERVICES, self.REQUESTS_ENTRY_NODES)
        self.heu_obj = WFCCRA(net_obj=self.net_obj, req_obj=self.req_obj, srv_obj=self.srv_obj, REQUESTED_SERVICES=self.REQUESTED_SERVICES, REQUESTS_ENTRY_NODES=self.REQUESTS_ENTRY_NODES)
        # self.random_obj = RCCRA(net_obj=self.net_obj, req_obj=self.req_obj, srv_obj=self.srv_obj, REQUESTED_SERVICES=self.REQUESTED_SERVICES, REQUESTS_ENTRY_NODES=self.REQUESTS_ENTRY_NODES)

    def get_state(self, entry_node=0, switch="none"):
        net_state = self.net_obj.get_state(entry_node, switch)
        # req_state = self.req_obj.get_state(assigned_nodes)
        # env_state = np.concatenate((req_state, net_state))
        env_state = net_state

        # normalized_env_state = scaler.fit_transform(env_state.reshape(-1, 1))
        # return normalized_env_state.reshape(1, -1)[0]

        return env_state

    def step(self, action, switch="none"):
        result = self.heu_obj.solve_per_req(action, switch)
        # optimum_result = self.model_obj.solve({}, switch, assigned_nodes)
        # accuracy = 0

        # print("*:   ", result["g"])

        if result["done"]:
            reward = 0
        else:
            max_cost = self.net_obj.MAX_COST_PER_TIER[result["pair"][0]][self.net_obj.get_tier_num(result["pair"][1])]
            min_cost = self.net_obj.MIN_COST_PER_TIER[result["pair"][0]][self.net_obj.get_tier_num(result["pair"][1])]
            tier_cost_range = max_cost - min_cost
            action_efficiency_range = 100
            action_efficiency = action_efficiency_range - (action_efficiency_range * (result["OF"] - min_cost) / tier_cost_range)
            reward_base = 100
            reward = reward_base ** (self.net_obj.get_tier_num(result["pair"][1]) + 1) + action_efficiency
            self.update_state(action, result)

        resulted_state = self.get_state(result["pair"][0], switch)

        return resulted_state, round(reward, 3), result["done"], result["info"], result["OF"], result["delay"]

    def update_state(self, action, result):
        self.net_obj.update_state(action, result, self.req_obj)
        # self.req_obj.update_state(action)

    def reset(self, SEED):
        # self.REQUESTS_ENTRY_NODES = specify_requests_entry_nodes(self.net_obj.FIRST_TIER_NODES, np.arange(self.NUM_REQUESTS), SEED)
        self.REQUESTS_ENTRY_NODES = np.zeros(self.NUM_REQUESTS).astype("int")
        self.req_obj = Request(NUM_REQUESTS=self.NUM_REQUESTS, NODES=self.net_obj.NODES, REQUESTS_ENTRY_NODES=self.REQUESTS_ENTRY_NODES, SEED=SEED)
        self.net_obj = Network(NUM_NODES=self.NUM_NODES, NUM_PRIORITY_LEVELS=self.NUM_PRIORITY_LEVELS)
        self.srv_obj = Service(NUM_SERVICES=self.NUM_SERVICES)
        self.REQUESTED_SERVICES = assign_requests_to_services(np.arange(self.NUM_SERVICES), np.arange(self.NUM_REQUESTS))
        self.heu_obj = WFCCRA(net_obj=self.net_obj, req_obj=self.req_obj, srv_obj=self.srv_obj, REQUESTED_SERVICES=self.REQUESTED_SERVICES, REQUESTS_ENTRY_NODES=self.REQUESTS_ENTRY_NODES)
        # self.random_obj = RCCRA(net_obj=self.net_obj, req_obj=self.req_obj, srv_obj=self.srv_obj, REQUESTED_SERVICES=self.REQUESTED_SERVICES, REQUESTS_ENTRY_NODES=self.REQUESTS_ENTRY_NODES)


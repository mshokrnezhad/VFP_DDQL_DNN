import math
import matplotlib.pyplot as plt
import numpy as np
rnd = np.random

# CAPACITY_REQUIREMENT_LB=10, CAPACITY_REQUIREMENT_UB=31, BW_REQUIREMENT_LB=20, BW_REQUIREMENT_UB=21, BURST_SIZE_LB=5, BURST_SIZE_UB=6 from v1 to v5
# CAPACITY_REQUIREMENT_LB=1, CAPACITY_REQUIREMENT_UB=4 for v6

class Request:
    def __init__(self, NUM_REQUESTS, NODES, REQUESTS_ENTRY_NODES, SEED=4, CAPACITY_REQUIREMENT_LB=4,
                 CAPACITY_REQUIREMENT_UB=8, BW_REQUIREMENT_LB=2, BW_REQUIREMENT_UB=3, DLY_REQUIREMENT_LB=10,
                 DLY_REQUIREMENT_UB=11, BURST_SIZE_LB=1, BURST_SIZE_UB=2
                 ):

        rnd.seed(SEED)
        self.NUM_REQUESTS = NUM_REQUESTS
        self.NODES = NODES
        self.REQUESTS_ENTRY_NODES = REQUESTS_ENTRY_NODES
        self.REQUESTS = np.arange(NUM_REQUESTS)
        self.CAPACITY_REQUIREMENT_LB = CAPACITY_REQUIREMENT_LB
        self.CAPACITY_REQUIREMENT_UB = CAPACITY_REQUIREMENT_UB
        self.BW_REQUIREMENT_LB = BW_REQUIREMENT_LB
        self.BW_REQUIREMENT_UB = BW_REQUIREMENT_UB
        self.DLY_REQUIREMENT_LB = DLY_REQUIREMENT_LB
        self.DLY_REQUIREMENT_UB = DLY_REQUIREMENT_UB
        self.BURST_SIZE_LB = BURST_SIZE_LB
        self.BURST_SIZE_UB = BURST_SIZE_UB
        self.CAPACITY_REQUIREMENTS = self.initialize_capacity_requirements()
        self.BW_REQUIREMENTS = self.initialize_bw_requirements()
        self.DELAY_REQUIREMENTS = self.initialize_delay_requirements()
        self.BURST_SIZES = self.initialize_burst_sizes()

    def initialize_capacity_requirements(self):
        capacity_requirements = np.array([rnd.randint(self.CAPACITY_REQUIREMENT_LB, self.CAPACITY_REQUIREMENT_UB)
                                          for i in self.REQUESTS])

        return capacity_requirements

    def initialize_bw_requirements(self):
        bw_requirements = np.array([rnd.randint(self.BW_REQUIREMENT_LB, self.BW_REQUIREMENT_UB)
                                    for i in self.REQUESTS])

        return bw_requirements

    def initialize_delay_requirements(self):
        delay_requirements = np.array([rnd.randint(self.DLY_REQUIREMENT_LB, self.DLY_REQUIREMENT_UB) for i in self.REQUESTS])

        return delay_requirements

    def initialize_burst_sizes(self):
        burst_sizes = np.array([rnd.randint(self.BURST_SIZE_LB, self.BURST_SIZE_UB) for i in self.REQUESTS])

        return burst_sizes

    """
    def get_state(self):
        active_requests = np.array([1 if i in self.REQUESTS else 0 for i in range(self.NUM_REQUESTS)])
        state = np.concatenate((active_requests, self.CAPACITY_REQUIREMENTS, self.BW_REQUIREMENTS,
                                self.DELAY_REQUIREMENTS, self.BURST_SIZES))

        return state
    """

    def get_state(self, switch="none", assigned_nodes=[]):
        state = []

        active_requests = np.array([1 if i in self.REQUESTS else 0 for i in range(self.NUM_REQUESTS)])
        per_node_capacity_requirements = np.array([np.sum((np.array(self.REQUESTS_ENTRY_NODES == i).astype(int)) * self.CAPACITY_REQUIREMENTS) for i in self.NODES])
        per_node_bw_requirements = np.array([np.sum((np.array(self.REQUESTS_ENTRY_NODES == i).astype(int)) * self.BW_REQUIREMENTS) for i in self.NODES])
        per_node_burst_sizes = np.array([np.sum((np.array(self.REQUESTS_ENTRY_NODES == i).astype(int)) * self.BURST_SIZES) for i in self.NODES])

        if switch == "srv_plc":
            state = np.concatenate((active_requests, per_node_capacity_requirements, per_node_bw_requirements, self.DELAY_REQUIREMENTS, per_node_burst_sizes))

        if switch == "pri_asg":
            per_destination_bw_requirements = np.array([np.sum((np.array(assigned_nodes == i).astype(int)) * self.BW_REQUIREMENTS) for i in self.NODES])
            state = np.concatenate((active_requests, per_node_capacity_requirements, per_node_bw_requirements, self.DELAY_REQUIREMENTS, per_node_burst_sizes, per_destination_bw_requirements))

        return state

    def update_state(self, action):
        # self.REQUESTS = np.delete(self.REQUESTS, action["req_id"])
        self.REQUESTS = np.setdiff1d(self.REQUESTS, [action["req_id"]])
        self.CAPACITY_REQUIREMENTS[action["req_id"]] = 0
        self.BW_REQUIREMENTS[action["req_id"]] = 0
        self.DELAY_REQUIREMENTS[action["req_id"]] = 0
        self.BURST_SIZES[action["req_id"]] = 0
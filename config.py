NETWORK_INIT = {  # The NETWORK class's default configuration. If you do not pass any attribute, it will default to the value defined here.
    "SEED": 4,
    "NUM_NODES": 3,
    "NUM_PRIORITY_LEVELS": 1,
    "NUM_TIERS": 3,
    "TIER_HEIGHT": 100,
    "TIER_WIDTH": 20,
    "DC_CAPACITY_MU": 100,  # Supposing that DC_CAPACITY has a normal distribution, DC_CAPACITY_MU is the mean.
    "DC_CAPACITY_SIGMA": 10,  # Representing the standard deviation of DC_CAPACITY supposing it has a normal distribution
    "DC_CAPACITY_GROWTH_RATE": 100,  # The mean DC_CAPACITY of nodes in tier i+1 is DC_CAPACITY_GROWTH_RATE greater than that of nodes in tier i.
    "DC_COST_MU": 1000,  # Supposing that DC_COST has a normal distribution, DC_COST_MU is the mean.
    "DC_COST_SIGMA": 10,  # Representing the standard deviation of DC_COST supposing it has a normal distribution.
    "DC_COST_DECREASE_RATE": 300,  # The mean DC_COST of nodes in tier i is DC_CAPACITY_GROWTH_RATE greater than that of nodes in tier i+1.
    "LINK_BW_MU": 250,  # Supposing that LINK_BW has a normal distribution, LINK_BW_MU is the mean.
    "LINK_BW_SIGMA": 50,  # Representing the standard deviation of LINK_BW supposing it has a normal distribution.
    "LINK_COST_MU": 20,  # Supposing that LINK_COST has a normal distribution, LINK_COST_MU is the mean. Note that LINK_COST dose not depend on the tier number of its nodes.
    "LINK_COST_SIGMA": 5,  # Representing the standard deviation of LINK_COST supposing it has a normal distribution.
    "BURST_SIZE_LIMIT": 200,
    "PACKET_SIZE": 1,
    "NUM_PATHS_UB": 1,  # Representing the number of paths considered betweeen each pair of nodes.
    "LINK_LENGTH_UB": 5,  # Representing the maximum length of paths. In other words, paths longer than LINK_LENGTH_UB will not be considered in the allocation procedure.
    "SAMPLE": "NET2"  # Representing the sample code. It should be one of the objects of NETWORK_SAMPLE.
}

NETWORK_SAMPLE = {
    "NET1": {
        "X_LOCS": [5, 10, 45, 35, 65, 70],
        "Y_LOCS": [55, 25, 65, 20, 55, 10],
        "LINKS_LIST": [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (1, 3), (3, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4)]
    },
    "NET2": {
        "X_LOCS": [5, 30, 65],
        "Y_LOCS": [55, 25, 65],
        "LINKS_LIST": [(0, 1), (1, 2)]
    },
    "NET3": {
        "X_LOCS": [5, 30, 65],
        "Y_LOCS": [55, 25, 65],
        "LINKS_LIST": [(0, 1), (1, 2)],
        "DC_CAPACITIES": [100, 200, 300]
    }
}

SERVICE_SAMPLE = {
    "SRVSET1": {
        "NUM_SERVICES": 3,
        "SEED": 4,
        "DC_CAPACITY_REQUIREMENTS_MU": [8, 8, 8],  # Supposing that DC_CAPACITY_REQUIREMENT[s] for service s has a normal distribution, DC_CAPACITY_REQUIREMENT_MU[s] is the mean.
        "DC_CAPACITY_REQUIREMENTS_SIGMA": [2, 2, 2],  # Representing the standard deviation of DC_CAPACITY_REQUIREMENT[s] for service s supposing it has a normal distribution.
        "BW_REQUIREMENTS_MU": [3, 3, 3],  # Supposing that BW_REQUIREMENT[s] for service s has a normal distribution, BW_REQUIREMENT_MU[s] is the mean.
        "BW_REQUIREMENTS_SIGMA": [1, 1, 1],  # Representing the standard deviation of BW_REQUIREMENT[s] for service s supposing it has a normal distribution.
        "DLY_REQUIREMENTS_MU": [10, 1, 100],  # Supposing that DLY_REQUIREMENT[s] for service s has a normal distribution, DLY_REQUIREMENT_MU[s] is the mean.
        "DLY_REQUIREMENTS_SIGMA": [1, 0, 1],  # Representing the standard deviation of DLY_REQUIREMENT[s] for service s supposing it has a normal distribution.
    },
    "SRVSET2": {
        "NUM_SERVICES": 1,
        "SEED": 4,
        "DC_CAPACITY_REQUIREMENTS_MU": [150],
        "DC_CAPACITY_REQUIREMENTS_SIGMA": [0],
        "BW_REQUIREMENTS_MU": [10],
        "BW_REQUIREMENTS_SIGMA": [0],
        "DLY_REQUIREMENTS_MU": [10],
        "DLY_REQUIREMENTS_SIGMA": [0],
    }
}

REQUEST_INIT = {  # The REQUEST class's default configuration. If you do not pass optional attributes, they will default to the values defined here.
    "SRV_OBJ": "",  # MANDATORY. An object of the service class should be passed to this class to define service-related parameters and requirements.
    "NET_OBJ": "",  # MANDATORY. An object of the network class should be passed to this class to define ENTRY_NODES.
    "NUM_REQUESTS": 1,
    "SEED": 4,
    "BURST_SIZE_MU": 2,  # Supposing that BURST_SIZE has a normal distribution, BURST_SIZE_MU is the mean.
    "BURST_SIZE_SIGMA": 1,  # Representing the standard deviation of BURST_SIZE supposing it has a normal distribution.
    "SAMPLE": ""  # Representing the sample code. It should be one of the objects of REQUEST_SAMPLE.
}

REQUEST_SAMPLE = {  # Do you want some parameters to be set in a specific way? Create a sample and send its code to INPUT.
    "REQ1": {
        "ENTRY_NODES": [0],
        "ASSIGNED_SERVICES": [0],
    }
}

AGENT_INIT = {  # The NETWORK class's default configuration. If you do not pass any attribute, it will default to the value defined here.
    "NUM_ACTIONS": 0,
    "INPUT_SHAPE": 0,
    "NAME": "",
    "EPSILON": 1, # Epsilon is initialized for the epsilon-greedy technique.
    "EPSILON_MIN": 0.05, # Setting the minimum bound for Epsilon.
    "EPSILON_DEC": 5e-4, # Defining a reduction factor for Epsilon.
    "GAMMA": 0.99, 
    "LR": 0.0001, # defining the learning rate
    "MEMORY_SIZE": 50000,  
    "BATCH_SIZE": 32,  
    "REPLACE_COUNTER": 1000,
    "CHECKPOINT_DIR": 'models/'
}

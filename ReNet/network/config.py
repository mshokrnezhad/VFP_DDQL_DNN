NETWORK_INIT = {  # The NETWORK class's default configuration. If you do not pass any attribute, it will default to the value defined here.  
    "SEED": 4,
    "NUM_NODES": 6,
    "NUM_PRIORITY_LEVELS": 1,
    "NUM_TIERS": 3,
    "TIER_HEIGHT": 100,
    "TIER_WIDTH": 20,
    "DC_CAPACITY_MU": 100,  # Supposing that DC_CAPACITY has a normal distribution, DC_CAPACITY_MU is the mean.
    "DC_CAPACITY_SIGMA": 10,  # Representing the standard deviation of DC_CAPACITY supposing it has a normal distribution
    "DC_CAPACITY_GROWTH_RATE": 100,  # The mean DC_CAPACITY of nodes in tier i+1 is DC_CAPACITY_GROWTH_RATE greater than that of nodes in tier i.
    "DC_COST_MU": 1000,  # Supposing that DC_COST has a normal distribution, DC_COST_MU is the mean.
    "DC_COST_SIGMA": 10,  # Representing the standard deviation of DC_COST supposing it has a normal distribution.
    "DC_COST_DECREASE_RATE": 200,  # The mean DC_COST of nodes in tier i is DC_CAPACITY_GROWTH_RATE greater than that of nodes in tier i+1.
    "LINK_BW_MU": 250,  # Supposing that LINK_BW has a normal distribution, LINK_BW_MU is the mean.
    "LINK_BW_SIGMA": 50,  # Representing the standard deviation of LINK_BW supposing it has a normal distribution.
    "LINK_COST_MU": 20,  # Supposing that LINK_COST has a normal distribution, LINK_COST_MU is the mean. Note that LINK_COST dose not depend on the tier number of its nodes.
    "LINK_COST_SIGMA": 5,  # Representing the standard deviation of LINK_COST supposing it has a normal distribution.
    "BURST_SIZE_LIMIT": 200,
    "PACKET_SIZE": 1,
    "NUM_PATHS_UB": 2,  # Representing the number of paths considered betweeen each pair of nodes.
    "LINK_LENGTH_UB": 5,  # Representing the maximum length of paths. In other words, paths longer than LINK_LENGTH_UB will not be considered in the allocation procedure.
    "SAMPLE": ""  # Representing the sample code. It should be one of the objects of NETWORK_SAMPLE.
}

NETWORK_SAMPLE = {
    "NET1": {
        "X_LOCS": [5, 10, 45, 35, 65, 70],
        "Y_LOCS": [55, 25, 65, 20, 55, 10],
        "LINKS_LIST": [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (1, 3), (3, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4)]
    }
}

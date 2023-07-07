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
    }
}

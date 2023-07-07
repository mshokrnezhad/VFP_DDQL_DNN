REQUEST_INIT = {  # The REQUEST class's default configuration. If you do not pass optional attributes, they will default to the values defined here.  
    "SRV_OBJ": "",  # MANDATORY. An object of the service class should be passed to this class to define service-related parameters and requirements.
    "NET_OBJ": "",  # MANDATORY. An object of the network class should be passed to this class to define ENTRY_NODES.
    "NUM_REQUESTS": 10,
    "SEED": 4,
    "BURST_SIZE_MU": 2,  # Supposing that BURST_SIZE has a normal distribution, BURST_SIZE_MU is the mean.
    "BURST_SIZE_SIGMA": 1,  # Representing the standard deviation of BURST_SIZE supposing it has a normal distribution.
    "SAMPLE": ""  # Representing the sample code. It should be one of the objects of REQUEST_SAMPLE.
}

REQUEST_SAMPLE = {  # Do you want some parameters to be set in a specific way? Create a sample and send its code to INPUT.
    "REQ1": {
        "ENTRY_NODES": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "ASSIGNED_SERVICES": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
}

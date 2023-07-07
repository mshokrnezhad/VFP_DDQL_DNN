import config
import numpy as np
rnd = np.random
REQUEST_INIT = config.REQUEST_INIT  # Default configs
REQUEST_SAMPLE = config.REQUEST_SAMPLE  # Sample requests


class Request:
    def __init__(self, INPUT):
        # Building INPUT
        self.INPUT = {key: INPUT[key] if key in INPUT else REQUEST_INIT[key] for key in REQUEST_INIT.keys()}
        # Defining base variables
        self.SEED = self.INPUT["SEED"]
        self.NUM_REQUESTS = self.INPUT["NUM_REQUESTS"]
        self.NET_OBJ = self.INPUT["NET_OBJ"]
        self.FIRST_TIER_NODES = self.NET_OBJ.get_first_tier_nodes()
        self.SRV_OBJ = self.INPUT["SRV_OBJ"]
        self.SERVICES = self.SRV_OBJ.SERVICES
        self.DC_CAPACITY_REQUIREMENTS_MU = self.SRV_OBJ.DC_CAPACITY_REQUIREMENTS_MU
        self.DC_CAPACITY_REQUIREMENTS_SIGMA = self.SRV_OBJ.DC_CAPACITY_REQUIREMENTS_SIGMA
        self.BW_REQUIREMENTS_MU = self.SRV_OBJ.BW_REQUIREMENTS_MU
        self.BW_REQUIREMENTS_SIGMA = self.SRV_OBJ.BW_REQUIREMENTS_SIGMA
        self.DLY_REQUIREMENTS_MU = self.SRV_OBJ.DLY_REQUIREMENTS_MU
        self.DLY_REQUIREMENTS_SIGMA = self.SRV_OBJ.DLY_REQUIREMENTS_SIGMA
        self.BURST_SIZE_MU = self.INPUT["BURST_SIZE_MU"]
        self.BURST_SIZE_SIGMA = self.INPUT["BURST_SIZE_SIGMA"]
        self.SAMPLE = self.INPUT["SAMPLE"]
        # Defining complementary variables
        rnd.seed(self.SEED)
        self.REQUESTS = np.arange(self.NUM_REQUESTS)
        self.ENTRY_NODES = self.initialize_entry_nodes()
        self.ASSIGNED_SERVICES = self.assign_services()
        self.DC_CAPACITY_REQUIREMENTS = self.initialize_capacity_requirements()
        self.BW_REQUIREMENTS = self.initialize_bw_requirements()
        self.DELAY_REQUIREMENTS = self.initialize_delay_requirements()
        self.BURST_SIZES = self.initialize_burst_sizes()

    def initialize_entry_nodes(self):
        if self.SAMPLE == "":
            return np.random.choice(self.FIRST_TIER_NODES, size=self.NUM_REQUESTS)
        else:
            return np.array(REQUEST_SAMPLE[self.SAMPLE]["ENTRY_NODES"])

    def assign_services(self):
        if self.SAMPLE == "":
            return np.random.choice(self.SERVICES, size=self.NUM_REQUESTS)
        else:
            return np.array(REQUEST_SAMPLE[self.SAMPLE]["ASSIGNED_SERVICES"]).astype(int)

    def initialize_capacity_requirements(self):
        mu = self.DC_CAPACITY_REQUIREMENTS_MU
        sigma = self.DC_CAPACITY_REQUIREMENTS_SIGMA
        srvs = self.ASSIGNED_SERVICES

        dc_capacity_requirements = np.array([np.random.normal(mu[srvs[r]], sigma[srvs[r]], 1)[0].round(0).astype(int) for r in self.REQUESTS])

        return dc_capacity_requirements

    def initialize_bw_requirements(self):
        mu = self.BW_REQUIREMENTS_MU
        sigma = self.BW_REQUIREMENTS_SIGMA
        srvs = self.ASSIGNED_SERVICES

        bw_requirements = np.array([np.random.normal(mu[srvs[r]], sigma[srvs[r]], 1)[0].round(0).astype(int) for r in self.REQUESTS])

        return bw_requirements

    def initialize_delay_requirements(self):
        mu = self.DLY_REQUIREMENTS_MU
        sigma = self.DLY_REQUIREMENTS_SIGMA
        srvs = self.ASSIGNED_SERVICES

        delay_requirements = np.array([np.random.normal(mu[srvs[r]], sigma[srvs[r]], 1)[0].round(0).astype(int) for r in self.REQUESTS])

        return delay_requirements

    def initialize_burst_sizes(self):
        mu = self.BURST_SIZE_MU
        sigma = self.BURST_SIZE_SIGMA

        burst_sizes = np.array([np.random.normal(mu, sigma, 1)[0].round(0).astype(int) for r in self.REQUESTS])

        return burst_sizes

    def get_state(self):
        request_features = []

        for r in self.REQUESTS:
            request_feature = []
            request_feature.append(self.ASSIGNED_SERVICES[r])
            request_feature.append(self.ENTRY_NODES[r])
            request_feature.append(self.DC_CAPACITY_REQUIREMENTS[r])
            request_feature.append(self.BW_REQUIREMENTS[r])
            request_feature.append(self.DELAY_REQUIREMENTS[r])
            request_features.append(request_feature)

        return request_features

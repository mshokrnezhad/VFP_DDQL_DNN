import config
import numpy as np
rnd = np.random
SERVICE_SAMPLE = config.SERVICE_SAMPLE  # Sample services


class Service:
    def __init__(self, INPUT):
        # Building INPUT
        self.INPUT = {key: SERVICE_SAMPLE[INPUT["SAMPLE"]][key] for key in SERVICE_SAMPLE[INPUT["SAMPLE"]].keys()}
        # Defining base variables
        self.SEED = self.INPUT["SEED"]
        self.NUM_SERVICES = self.INPUT["NUM_SERVICES"]
        self.DC_CAPACITY_REQUIREMENTS_MU = self.INPUT["DC_CAPACITY_REQUIREMENTS_MU"]
        self.DC_CAPACITY_REQUIREMENTS_SIGMA = self.INPUT["DC_CAPACITY_REQUIREMENTS_SIGMA"]
        self.BW_REQUIREMENTS_MU = self.INPUT["BW_REQUIREMENTS_MU"]
        self.BW_REQUIREMENTS_SIGMA = self.INPUT["BW_REQUIREMENTS_SIGMA"]
        self.DLY_REQUIREMENTS_MU = self.INPUT["DLY_REQUIREMENTS_MU"]
        self.DLY_REQUIREMENTS_SIGMA = self.INPUT["DLY_REQUIREMENTS_SIGMA"]
        # Defining complementary variables
        rnd.seed(self.SEED)
        self.SERVICES = np.arange(self.NUM_SERVICES)

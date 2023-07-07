import math
import matplotlib.pyplot as plt
import numpy as np
rnd = np.random


class Service:
    def __init__(self, NUM_SERVICES, SEED=4):
        self.NUM_SERVICES = NUM_SERVICES
        self.SERVICES = np.arange(NUM_SERVICES)
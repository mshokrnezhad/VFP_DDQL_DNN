from network.Network import Network
from request.Request import Request
from service.Service import Service
import numpy as np


class Environment:
    def __init__(self, INPUT):
        self.INPUT = INPUT
        self.NET_INPUT = self.INPUT["NET"]
        self.SRV_INPUT = self.INPUT["SRV"]
        self.REQ_INPUT = self.INPUT["REQ"]
        self.NET_OBJ = Network(self.NET_INPUT)
        self.SRV_OBJ = Service(self.SRV_INPUT)
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

    def reset(self):
        self.NET_INPUT = self.INPUT["NET"]
        self.SRV_INPUT = self.INPUT["SRV"]
        self.REQ_INPUT = self.INPUT["REQ"]
        self.NET_OBJ = Network(self.NET_INPUT)
        self.SRV_OBJ = Service(self.SRV_INPUT)
        self.REQ_INPUT["NET_OBJ"] = self.NET_OBJ
        self.REQ_INPUT["SRV_OBJ"] = self.SRV_OBJ
        self.REQ_OBJ = Request(self.REQ_INPUT)

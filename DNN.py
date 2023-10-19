import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch as T
import os
import numpy as np


class DNN(nn.Module):
    def __init__(self, INPUT):
        super().__init__()
        # Building INPUT
        self.INPUT = INPUT
        # Defining base variables
        self.CHECKPOINT_DIR = self.INPUT["CHECKPOINT_DIR"]
        self.CHECKPOINT_FILE = os.path.join(self.CHECKPOINT_DIR, self.INPUT["NAME"])
        self.NUM_LAYERS = self.INPUT["NUM_LAYERS"]
        self.SIZE_LAYERS = self.INPUT["SIZE_LAYERS"]
        self.fc = self.generate_layers_v01()
        self.optimizer = opt.Adam(self.parameters(), lr=self.INPUT["LR"])
        self.criterion = nn.MSELoss()

    def generate_layers_v01(self):
        fc = nn.ModuleList()

        for l in range(self.NUM_LAYERS+1):
            fc.append(nn.Linear(self.SIZE_LAYERS[l], self.SIZE_LAYERS[l+1]))

        return fc

    def forward(self, state):  # forward propagation includes defining layers
        out = []

        out.append(F.relu(self.fc[0](state)))
        for l in range(1, self.NUM_LAYERS):
            out.append(F.relu(self.fc[l](out[l-1])))
        out.append(self.fc[-1](out[self.NUM_LAYERS-1]))

        return out[-1]

    def save_checkpoint(self):
        print(f'Saving {self.CHECKPOINT_FILE}...')
        T.save(self.state_dict(), self.CHECKPOINT_FILE)

    def load_checkpoint(self):
        print(f'Loading {self.CHECKPOINT_FILE}...')
        self.load_state_dict(T.load(self.CHECKPOINT_FILE))

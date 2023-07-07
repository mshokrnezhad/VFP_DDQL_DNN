import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch as T
import os
import numpy as np


class DNN(nn.Module):
    def __init__(self, LR, NUM_ACTIONS, INPUT_SHAPE, NAME, CHECKPOINT_DIR, H1=256, H2=128, H3=64, H4=32):
        super().__init__()
        self.CHECKPOINT_DIR = CHECKPOINT_DIR
        self.CHECKPOINT_FILE = os.path.join(self.CHECKPOINT_DIR, NAME)
        self.fc1 = nn.Linear(INPUT_SHAPE, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, H3)
        self.fc4 = nn.Linear(H3, H4)
        self.fc5 = nn.Linear(H4, NUM_ACTIONS)
        self.optimizer = opt.Adam(self.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def forward(self, state):  # forward propagation includes defining layers
        out1 = F.relu(self.fc1(state))
        out2 = F.relu(self.fc2(out1))
        out3 = F.relu(self.fc3(out2))
        out4 = F.relu(self.fc4(out3))
        out5 = self.fc5(out4)

        return out5

    def save_checkpoint(self):
        print(f'Saving {self.CHECKPOINT_FILE}...')
        T.save(self.state_dict(), self.CHECKPOINT_FILE)

    def load_checkpoint(self):
        print(f'Loading {self.CHECKPOINT_FILE}...')
        self.load_state_dict(T.load(self.CHECKPOINT_FILE))

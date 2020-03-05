
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.grid import Grid
import numpy as np
import copy 
import random

# Config for the MLP. This is used to change the size of the MLP,
# including the dimensions of its input and output layers, to
# accommodate boards of different sizes.
class MLPConfig:
    def __init__(self, board_size, num_colors):
        self.board_size = board_size
        self.num_colors = num_colors
        self.num_hidden = 5
        self.nodes_per_layer = 5
        self.num_actions = num_colors * 4
        self.exploration_rate = 0.1


# This MLP is used as the value function approximation (VFA) for
# the Q function. The input is a feature representation of a state
# and the output is a softmax over all possible actions.
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        self.config = config
        self.layers = []
        input_size = config.board_size * config.board_size + 2*config.num_colors
        for h in range(config.num_hidden):
        	self.layers.append(nn.Linear(input_size, config.nodes_per_layer))
        	input_size = config.nodes_per_layer
        self.output = nn.Linear(input_size, config.num_actions)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        for h in range(self.config.num_hidden):
            x = self.layers[h](x)
            x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

    def get_Q(self, state):
        return self(state.float())

    def get_next_action(self, state):
        if random.random() > self.config.exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        _, index_tensor = self.get_Q(state).max(0)
        index = index_tensor.item()
        return self.convert_index_to_tuple(index), index

    def random_action(self):
        index = random.randrange(0, self.config.num_actions)
        return self.convert_index_to_tuple(index), index

    def convert_index_to_tuple(self, index):
        color = int(index / 4 + 1)
        direction = int(index % 4)
        return (color, direction)

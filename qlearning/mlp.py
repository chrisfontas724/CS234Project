
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
    def __init__(self, board_size, num_colors, num_hidden=5):
        self.board_size = board_size
        self.num_colors = num_colors
        self.num_hidden = num_hidden
        self.feature_vector_size = (self.board_size **2) + 6*self.num_colors + 2
        self.nodes_per_layer = self.feature_vector_size * 2
        self.num_actions = num_colors * 4


# This MLP is used as the value function approximation (VFA) for
# the Q function. The input is a feature representation of a state
# and the output is a softmax over all possible actions.
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        self.config = config
        self.layers = []
        input_size = self.config.feature_vector_size
        for h in range(config.num_hidden):
        	self.layers.append(nn.Linear(input_size, config.nodes_per_layer))
        	input_size = config.nodes_per_layer
        self.output = nn.Linear(input_size, config.num_actions)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        for h in range(self.config.num_hidden):
            x = self.layers[h](x)
            x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

    def get_Q(self, state):
        return self(state.float())

    def get_next_action(self, state, grad, exploration_rate):
        if random.random() > exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state, grad)
        else:
            return self.random_action(state, grad)

    def greedy_action(self, state, grad):
        if grad:
            value, index_tensor = self.get_Q(state).max(0)
            index = index_tensor.item()
            return self.convert_index_to_tuple(index), value
        with torch.no_grad():
            value, index_tensor = self.get_Q(state).max(0)
            index = index_tensor.item()
            return self.convert_index_to_tuple(index), value

    def random_action(self, state, grad):
        if grad:
            index = random.randrange(0, self.config.num_actions)
            return self.convert_index_to_tuple(index), self.get_Q(state)[index]   
        with torch.no_grad():
            index = random.randrange(0, self.config.num_actions)
            return self.convert_index_to_tuple(index), self.get_Q(state)[index]

    def convert_index_to_tuple(self, index):
        color = int(index / 4 + 1)
        direction = int(index % 4)
        return (color, direction)

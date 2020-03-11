
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
        self.feature_vector_size = (self.board_size **2) + 6*self.num_colors + 4
        self.nodes_per_layer = self.feature_vector_size 
        self.num_actions = num_colors * 4


# This MLP is used as the value function approximation (VFA) for
# the Q function. The input is a feature representation of a state
# and the output is a softmax over all possible actions.
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        #print("Init MLP: ")
        self.config = config
        self.layers = []
        input_size = self.config.feature_vector_size
        node_count = config.nodes_per_layer

        self.layer1 = nn.Linear(input_size, node_count, bias=True)
        torch.nn.init.xavier_uniform(self.layer1.weight)

        input_size = node_count
        node_count -= 2
        self.layer2 = nn.Linear(input_size, node_count, bias=True)
        torch.nn.init.xavier_uniform(self.layer2.weight)

        input_size = node_count
        node_count -= 2
        self.layer3 = nn.Linear(input_size, node_count, bias=True)
        torch.nn.init.xavier_uniform(self.layer3.weight)

        # input_size = node_count
        # node_count -= 2
        # self.layer4 = nn.Linear(input_size, node_count, bias=True)
        # torch.nn.init.xavier_uniform(self.layer4.weight)

        # input_size = node_count
        # node_count -= 2
        # self.layer5 = nn.Linear(input_size, node_count, bias=True)
        # torch.nn.init.xavier_uniform(self.layer5.weight)

        # input_size = node_count
        # node_count -= 2
        # self.layer6 = nn.Linear(input_size, node_count, bias=True)
        # torch.nn.init.xavier_uniform(self.layer6.weight)

        # input_size = node_count
        # node_count -= 2
        # self.layer7 = nn.Linear(input_size, node_count, bias=True)
        # torch.nn.init.xavier_uniform(self.layer7.weight)

        input_size = node_count
        node_count -= 2
        self.layer8 = nn.Linear(input_size, node_count, bias=True)
        torch.nn.init.xavier_uniform(self.layer8.weight)

        input_size = node_count
        node_count -= 2
        self.layer9 = nn.Linear(input_size, node_count, bias=True)
        torch.nn.init.xavier_uniform(self.layer9.weight)

        input_size = node_count
        node_count -= 2
        self.layer10 = nn.Linear(input_size, node_count, bias=True)
        torch.nn.init.xavier_uniform(self.layer10.weight)


        input_size = node_count
        node_count = config.num_actions
        self.output = nn.Linear(input_size, node_count, bias=False)
        torch.nn.init.xavier_uniform(self.output.weight)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.drop_layer = nn.Dropout(p=0.1)


    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)
        x = self.drop_layer(x)

        # x = self.layer4(x)
        # x = self.relu(x)

        # x = self.layer5(x)
        # x = self.relu(x)
        # x = self.drop_layer(x)

        # x = self.layer6(x)
        # x = self.relu(x)

        # x = self.layer7(x)
        # x = self.sigmoid(x)

        x = self.layer8(x)
        x = self.sigmoid(x)

        x = self.layer9(x)
        x = self.sigmoid(x)

        x = self.layer10(x)
        x = self.tanh(x)

        x = self.output(x)
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
            #print("Test: " + str(value) + " " + str(index_tensor.item()))
            index = index_tensor.item()
            action, q_sa = self.convert_index_to_tuple(index), value
           # print("Action: " + str(action) + " QSA: " + str(q_sa))
            return action, q_sa

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

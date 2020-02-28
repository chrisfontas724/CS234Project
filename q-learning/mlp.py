
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.grid import Grid
import numpy as np
import copy 

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

class MLP(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.config = config
        self.layers = []
        input_size = config.board_size * config.board_size + 2*config.num_colors
        for h in range(config.num_hidden):
        	self.layers.append(nn.Linear(input_size, config.nodes_per_layer))
        	input_size = config.nodes_per_layer
        self.output = nn.Linear(input_size, config.num_actions)

        self.relu = nn.ReLU()
    	self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
    	for h in range(self.config.num_hidden):
    		x = self.layers[h](x)
    		x = self.sigmoid(x)
    	x = self.output(x)
    	x = self.softmax(x)

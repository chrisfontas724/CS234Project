from models.grid import Grid
from qlearning.qstate import QState
from qlearning.mlp import MLPConfig, MLP
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from renderer.renderer import GridRenderer
import copy
import torch.nn.functional as F
import random
from optparse import OptionParser
from matplotlib import pyplot as plt


max_items_in_replay = 100000

def load_grids(size):
	print("Loading grids....")
	grids = list()
	for i in range(1, 900):
		grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_" + str(i) + ".txt")
		grids.append(grid)
	return grids

def initialize_replay_buffer_with_single_grid(size):
	pass

def get_mini_batch(replay_buffer, num_samples=50):
	# To preserve the order of the list, you could do:
	randIndex = random.sample(range(len(replay_buffer)), num_samples)
	randIndex.sort()
	return [replay_buffer[i] for i in randIndex]


# Train a tabular dqn model here.
def train(size, gamma=0.9):
	pass

# Play tabular here.
def play(mlp, size=4):
	pass

# Determines the board size we will be using for training.
def get_options():
	parser = OptionParser()

	parser.add_option("-s", "--size",
						action="store", # optional because action defaults to "store"
                      	dest="size",
                      	default=5,
                      	help="Size of board to use",)

	return parser.parse_args()

def main():
	# Grab the command line options.
	options, args = get_options()
	print("Training with boards of size ", options.size)

if __name__ == "__main__":
	main()
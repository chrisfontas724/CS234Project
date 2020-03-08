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


starting_items_in_replay = 15000
max_items_in_replay = 10000000

def load_grids(size):
	print("Loading grids....")
	grids = list()
	for i in range(1, 900):
		grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_" + str(i) + ".txt")
		grids.append(grid)
	return grids

def initialize_replay_buffer(grids, mlp):
	print("Initializing replay buffer...")
	result = set()
	for grid in grids:
		state = QState(grid.start_state)
		action, q_sa = mlp.get_next_action(state.get_feature_vector(), grad=True, exploration_rate = 1.0)
		new_state, reward, terminal = state.step(action)
		sars = (state, action, reward, q_sa, new_state, terminal)
		result.add(sars)
	return result

# Hacky/test function to see if we can get DQN working with just a single board, instead
# of with the entire training set.
def initialize_replay_buffer_with_single_grid(size, mlp):
	print("Initialize replay buffer with single board")
	result = set()
	grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_100.txt")
	state = QState(grid.start_state)
	for i in range(starting_items_in_replay):
		action, q_sa = mlp.get_next_action(state.get_feature_vector(), grad=True, exploration_rate=1.0)
		new_state, reward, terminal = state.step(action)
		sars = (state, action, reward, q_sa, new_state, terminal)
		result.add(sars)
		state = new_state if not terminal else QState(grid.start_state)
	return result

# Randomly select |num_samples| sars tuples from the replay buffer.
def get_mini_batch(replay_buffer, num_samples=50):
	return random.sample(replay_buffer, num_samples)

# Copy over the input training network to the target network by doing
# a deep copy over its entire state dict.
def update_target_network(mlp):
	target_mlp = MLP(mlp.config)
	target_mlp.load_state_dict(copy.deepcopy(mlp.state_dict()))
	target_mlp.train(False)
	return target_mlp

def make_mlp(size, cols):
	# Create the MLP network with the configuration.
	mlp_config = MLPConfig(size, cols,  num_hidden=6)
	mlp = MLP(mlp_config)
	mlp = mlp.float()
	return mlp

def train(size, gamma=0.9):
	mlp = make_mlp(size, size-1)
	replay_buffer = initialize_replay_buffer_with_single_grid(size, mlp)  #initialize_replay_buffer(load_grids(size), mlp)
	epsilon = 1.0
	optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01) # optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)
	loss_function = torch.nn.MSELoss()


	target_mlp = None
	update_target = 1000
	train_steps = 10000000
	average_loss = 0.
	target_state_dict = None
	losses = list()
	for i in range(train_steps):
		if i%10000==0:
			if epsilon > 0.1 :
				epsilon -=9e-7
		# Update the target nextwork to match the update network.
		if i % update_target == 0:
			print("Update target network!")
			target_mlp = update_target_network(mlp)
			target_state_dict = copy.deepcopy(target_mlp.state_dict())

		# Sanity check to make sure we're not accidentally modifying the
		# target network in between updates.
		for key,item in target_state_dict.items():
			if not torch.all(torch.eq(item, target_mlp.state_dict()[key])):
				print("Target has been altered!")

		# Sample a random mini-batch from the replay buffer.
		batch = get_mini_batch(replay_buffer, 200)
		target_values = torch.zeros([len(batch), 1], dtype=torch.float32)
		model_values = torch.zeros([len(batch), 1], dtype=torch.float32)

		for b in range(len(batch)):
			# Expand out the tuple.
			state, action, reward, q_sa, new_state, terminal = batch[b]

			# Get the maximum action for q(s',a'; w-). If we're
			# in the terminal/winning state, then there is no next
			# state, and so q_prime_sa is just 0.
			_, q_prime_sa = target_mlp.greedy_action(new_state.get_feature_vector(), grad=False) if terminal else None, torch.tensor(0.)

			target_values[b] = reward + gamma*q_prime_sa
			model_values[b] = q_sa

			# Update the entries in the replay buffer. Don't bother adding winning states in
			# because they have no viable actions.
			if not terminal:
				updated_action, updated_qsa = mlp.get_next_action(new_state.get_feature_vector(), grad=True,exploration_rate=epsilon)
				updated_state, reward, terminal = new_state.step(action)
				replay_buffer.add((new_state, updated_action, reward, updated_qsa, updated_state, terminal))

		# Get average loss
		loss = loss_function(target_values, model_values)


		# Remove the oldest elements from the replay buffer.
		if len(replay_buffer) > max_items_in_replay:
			replay_buffer = replay_buffer[len(replay_buffer)-max_items_in_replay:]

		# Perform gradient descent.
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		for param in mlp.parameters():
			param.grad.data.clamp_(-1, 1)
		
		optimizer.step()
		average_loss += loss.item()


		# Print out loss calculations.		
		if i % 10 == 0:
			print("Iteration " + str(i) + " average loss: " + str(average_loss / 10))
			losses.append(average_loss / 10)
			average_loss = 0.0

		if i % 10000 == 0:
			# test_play = play(mlp, size)
			# print("Test won!") if test_play else print("Test lost!")
			torch.save(mlp.state_dict(), "q_models/" + str(size) + "x" + str(size) + "_" + str(i) + "_" + "model_v2.txt")


	plt.plot(losses)
	plt.show()
	return mlp


def play(mlp, size=4):
	grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_950.txt")
	#renderer = GridRenderer("Q-Learning")

	# Wrap the states as QStates to get functionality
	# specifically needed for Q-learning.
	state = QState(grid.start_state)
	won = False
	while True:
    	# Grab the feature vector for the given QState.
		features = state.get_feature_vector()

		# Get best action from the MLP.
		action, _ = mlp.greedy_action(features, grad=False)
		print("Take action: ", action)

		if not state.is_viable_action(action):
			break

     	# Advance to the next state.
		state = state.next_state(action)

		# Break if we're in the winning state.
		if state.is_winning():
			won = True
			break
	#
	#renderer.render(state.state)
	#renderer.tear_down()

	return won


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

	mlp = train(int(options.size))
	torch.save(mlp.state_dict(), "q_models/model.txt")

	status = play(mlp, int(options.size))
	print("We " + ("won \\^_^/" if status else "lost =("))

if __name__ == "__main__":
	main()
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

max_items_in_replay = 10000


def load_grids(size):
	print("Loading grids....")
	grids = list()
	for i in range(1, 900):
		grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_" + str(i) + ".txt")
		grids.append(grid)
	return grids


def initialize_replay_buffer(grids, mlp):
	print("Initializing replay buffer...")
	result = list()
	for grid in grids:
		state = QState(grid.start_state)
		action, q_sa = mlp.get_next_action(state.get_feature_vector(), grad=True)
		new_state, reward, terminal = state.step(action)
		sars = (state, action, reward, q_sa, new_state, terminal)
		result.append(sars)
	return result



def get_mini_batch(replay_buffer, num_samples=50):
	# To preserve the order of the list, you could do:
	randIndex = random.sample(range(len(replay_buffer)), num_samples)
	randIndex.sort()
	return [replay_buffer[i] for i in randIndex]

def update_target_network(mlp):
	target_mlp = MLP(mlp.config)
	target_mlp.load_state_dict(copy.deepcopy(mlp.state_dict()))
	target_mlp.train(False)
	return target_mlp

def make_mlp(size, cols):
	# Create the MLP network with the configuration.
	mlp_config = MLPConfig(size, cols, 30)
	mlp = MLP(mlp_config)
	mlp = mlp.float()
	return mlp


def train(size, gamma=0.9):
	mlp = make_mlp(4,3)
	replay_buffer = initialize_replay_buffer(load_grids(4), mlp)

	optimizer = optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)
	loss_function = torch.nn.MSELoss()

	target_mlp = None
	update_target = 100
	train_steps = 10000000
	average_loss = 0.
	for i in range(train_steps):
		# Update the target nextwork to match the update network.
		if i % update_target == 0:
			target_mlp = update_target_network(mlp)


		# Sample a random mini-batch from the replay buffer.
		batch = get_mini_batch(replay_buffer)
		loss = torch.tensor(0.)
		for sars in batch:
			# Expand out the tuple.
			state, action, reward, q_sa, new_state, terminal = sars

			# Get the maximum action for q(s',a'; w-). If we're
			# in the terminal/winning state, then there is no next
			# state, and so q_prime_sa is just 0.
			_, q_prime_sa = target_mlp.greedy_action(new_state.get_feature_vector(), grad=False) if terminal else None, torch.tensor(0.)

			# Update the  loss.
			loss += loss_function(reward + gamma*q_prime_sa, q_sa)

			# Update the entries in the replay buffer. Don't bother adding winning states in
			# because they have no viable actions.
			if not new_state.is_winning():
				updated_action, updated_qsa = mlp.get_next_action(new_state.get_feature_vector(), grad=True)
				updated_state, reward, terminal = new_state.step(action)
				replay_buffer.append((new_state, updated_action, reward, updated_qsa, updated_state, terminal))

		# Get average loss
		loss /= len(batch)

		# Remove the oldest elements from the replay buffer.
		replay_buffer = replay_buffer[len(batch):]

       	# Perform gradient descent.
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()
		average_loss += loss.item()

		# Print out loss calculations.		
		if i % 10 == 0:
			print("Iteration " + str(i) + " average loss: " + str(average_loss / 10))
			average_loss = 0.0

	return mlp


def play(mlp):
	grid = Grid(filename="levels/grid_1.txt")
	renderer = GridRenderer("Q-Learning")

	# Wrap the states as QStates to get functionality
	# specifically needed for Q-learning.
	state = QState(grid.start_state)
	won = False
	while True:
    	# Grab the feature vector for the given QState.
		features = state.get_feature_vector()

		# Get best action from the MLP.
		action, _ = mlp.greedy_action(features)
		print("Take action: ", action)

		if not state.is_viable_action(action):
			break

     	# Advance to the next state.
		state = state.next_state(action)

		# Break if we're in the winning state.
		if state.is_winning():
			won = True
			break
	
	renderer.render(state.state)
	renderer.tear_down()

	return won

def main():
	mlp = train(4)
	torch.save(mlp.state_dict(), "q_models/model.txt")

	status = play(mlp)
	print("We " + ("won \\^_^/" if status else "lost =("))

if __name__ == "__main__":
	main()
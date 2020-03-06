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

def train(size, gamma=0.9):

	grid_idx = 1
    # Hardcode a simple grid for now.
	grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_" + str(grid_idx) + ".txt")

	# Create the MLP network with the configuration.
	mlp_config = MLPConfig(grid.size, grid.num_cols, 30)
	mlp = MLP(mlp_config)
	mlp = mlp.float()

	optimizer = optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)
	loss_function = torch.nn.MSELoss()
	state = QState(grid.start_state)

	target_mlp = None
	update_target = 10000
	train_steps = 10000000
	average_loss = 0.
	for i in range(train_steps):
		# Update the target nextwork to match the update network.
		if i % update_target == 0:
			target_mlp = MLP(mlp.config)
			target_mlp.load_state_dict(copy.deepcopy(mlp.state_dict()))
			target_mlp.train(False)

		# Grab the feature vectors for the current state.
		features = state.get_feature_vector()

		# Use e-greedy to get the next action.
		action, q_sa = mlp.get_next_action(features, grad=True)

		# Take that action and see what happens next.
		new_state, reward, terminal = state.step(action)

		# Get the maximum action for q(s',a'; w-). If we're
		# in the terminal/winning state, then there is no next
		# state, and so q_prime_sa is just 0.
		if terminal:
			q_prime_sa = torch.tensor(0.)
		else:
			_, q_prime_sa = target_mlp.greedy_action(new_state.get_feature_vector(), grad=False)

		# Calculate loss.
		loss = loss_function(reward + gamma*q_prime_sa, q_sa)

       	# zero the parameter gradients
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		average_loss += loss.item()

		# Update the current state. If we won, go back to the beginning.
		if not terminal:
			state = new_state
		# Start a new grid.
		else:
			print("Load new grid!")
			grid_idx += 1
			if grid_idx >= 900:
				grid_idx = 1
			grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_" + str(grid_idx) + ".txt")
			state = QState(grid.start_state)
		
		if i % 1000 == 0:
			print("Iteration " + str(i) + " average loss: " + str(average_loss / 1000))
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
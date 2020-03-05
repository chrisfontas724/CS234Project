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

def train(grid, mlp, gamma=0.9):
	optimizer = optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)
	loss_function = torch.nn.MSELoss()
	state = QState(grid.start_state)

	target_mlp = None
	update_target = 10000
	train_steps = 100000
	average_loss = 0.
	for i in range(train_steps):

		# Update the target nextwork to match the update network.
		if i % update_target == 0:
			target_mlp = MLP(mlp.config)
			target_mlp.load_state_dict(copy.deepcopy(mlp.state_dict()))
			target_mlp.train(False)

		# Grab the feature vectors for the current state.
		features = state.get_feature_vector()

		# Get the q values for the state from the update network.
		old_state_q_values = mlp(features.float())

		# Choose a random action for q(s,a; w)
		possible_actions = state.state.possible_actions()
		if len(possible_actions) == 0:
			state = QState(grid.start_state)
			continue

		action = random.choice(possible_actions)
		index = (action[0] - 1)*4 + action[1]
		q_sa = old_state_q_values[index]

		# Take that action and see what happens next.
		new_state, reward, terminal = state.step(action)
		# if terminal:
		# 	state = QState(grid.start_state)
		# 	continue

		# Use e-greedy algorithm to get q(s',a'; w-)
		new_features = new_state.get_feature_vector()
		new_action, q_prime_sa = target_mlp.get_next_action(new_features.float())


		# Calculate loss.
		loss = loss_function(reward + gamma*q_prime_sa, q_sa)

       	# zero the parameter gradients
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		average_loss += loss.item()

		# Update the current state.
		state = new_state
		
		if i % 1000 == 0:
			print("Iteration " + str(i) + " average loss: " + str(average_loss / 1000))
			average_loss = 0.0

	return mlp


def play(grid, mlp):

	renderer = GridRenderer("Q-Learning")

	# Wrap the states as QStates to get functionality
	# specifically needed for Q-learning.
	state = QState(grid.start_state)
	won = False
	while True:
    	# Grab the feature vector for the given QState.
		features = state.get_feature_vector()

		# Get best action from the MLP.
		action, _ = mlp.greedy_action(features.float())
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
    # Hardcode a simple grid for now.
	grid = Grid(filename="levels/grid_1.txt")

	# Create the MLP network with the configuration.
	mlp_config = MLPConfig(grid.size, grid.num_cols, 15)
	mlp = MLP(mlp_config)
	mlp = mlp.float()

	mlp = train(grid, mlp)

	status = play(grid, mlp)
	print("We " + ("won \\^_^/" if status else "lost =("))

if __name__ == "__main__":
	main()
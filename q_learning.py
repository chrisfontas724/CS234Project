from models.grid import Grid
from qlearning.qstate import QState
from qlearning.mlp import MLPConfig, MLP
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn


def train(grid, mlp, loss_function, optimizer):

	state = QState(grid.start_state)

	train_steps = 1000
	for _ in range(train_steps):
		features = state.get_feature_vector()

		old_state_q_values = mlp(features.float())

		action, index = mlp.get_next_action(features)

		new_state, reward, terminal = state.step(action)
		print("Result tuple: ", reward, terminal)

		new_features = new_state.get_feature_vector()

		new_state_q_values = mlp(new_features.float())
		_, new_index = new_state_q_values.max(0)

		loss = loss_function(reward + new_state_q_values[new_index], old_state_q_values[index])
		print("LOSS: ", loss)
		print("\n")

		loss.backward()
		optimizer.step()



def play(grid, mlp):

	# Wrap the states as QStates to get functionality
	# specifically needed for Q-learning.
	state = QState(grid.start_state)
	while True:
    	# Grab the feature vector for the given QState.
		features = state.get_feature_vector()

     	# Pass the features through the network to see
     	# what it gives as the best action to take.
		action_probabilities = mlp(features.float())

     	# Grab the index of the best action.
		_, index_tensor = action_probabilities.max(0)
		index = index_tensor.item()
		print("Index: ", index)

     	# Figure out what that corresponds too.
		color = int(index / 4 + 1)
		direction = int(index % 4)

		action = (color, direction)
		print("Take action: ", action)

		if not state.is_viable_action(action):
			return False

     	# Advance to the next state.
		state = state.next_state(action)
     
		# Break if we're in the winning state.
		if state.is_winning():
			return True

def main():
    # Hardcode a simple grid for now.
	grid = Grid(filename="levels/grid_1.txt")

	# Create the MLP network with the configuration.
	mlp_config = MLPConfig(grid.size, grid.num_cols)
	mlp = MLP(mlp_config)
	mlp = mlp.float()

	criterion = nn.MSELoss()
	optimizer = optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)
	train(grid, mlp, criterion, optimizer)

	status = play(grid, mlp)
	print("We " + ("won \\^_^/" if status else "lost =("))

if __name__ == "__main__":
	main()
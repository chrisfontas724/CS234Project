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
# import torch
# from torchvision import models
# from torchsummary import summary

starting_items_in_replay = 150000
max_items_in_replay = 100000000

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
		action, q_sa = mlp.get_next_action(state.get_feature_vector(), grad=False, exploration_rate = 1.0)
		new_state, reward, terminal = state.step(action)

		action_index = (action[0]-1)*4 + action[1]

		sars = (state, action_index, reward, new_state, terminal)
		result.add(sars)
	return list(result)

# Hacky/test function to see if we can get DQN working with just a single board, instead
# of with the entire training set.
def initialize_replay_buffer_with_single_grid(size, mlp):
	print("Initialize replay buffer with single board")
	result = set()
	grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_100.txt")
	state = QState(grid.start_state)
	for i in range(starting_items_in_replay):
		action, q_sa = mlp.get_next_action(state.get_feature_vector(), grad=False, exploration_rate=1.0)
		new_state, reward, terminal = state.step(action)
		sars = (state, action, reward, q_sa, new_state, terminal)
		result.add(sars)
		state = new_state if not terminal else QState(grid.start_state)
	return result

def initialize_buffer_with_all_tuples(size, mlp):
	grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_1.txt")
	result = set()
	states = grid.generate_all_states()
	for state in states:
		q_state = QState(state)

		# Don't store the winning state in the replay buffer.
		if q_state.is_winning():
			print("Don't include me!")
			continue

		for action in range((size-1)*4):

			color = int(action / 4 + 1)
			direction = int(action % 4)
			action_tu = (color, direction)


			new_state, reward, terminal = q_state.step(action_tu)
			sars = (q_state, action, reward, new_state, terminal)
			result.add(sars)
	print("Initial replay buffer size: ", len(result))
	return list(result)

def initialize_buffer_with_single_test_tuple(size):
	grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_100.txt")
	result = set()
	states = grid.generate_all_states()
	find = False
	print("Start state: ")
	print(grid.start_state.spaces)
	for state in states:
		q_state = QState(state)
		for action in range((size-1)*4):

			color = int(action / 4 + 1)
			direction = int(action % 4)
			action_tu = (color, direction)

			new_state, reward, terminal = q_state.step(action_tu)

			if terminal:
				sars = (q_state, action, reward, new_state, terminal)
				print("Old state: ")
				print(q_state.state.spaces)
				print("New state")
				print(new_state.state.spaces)

				print("Action: ", action_tu)
				print("Action index: ", action)
				print("Old state winning: ", q_state.is_winning())
				print("New state winning: ", new_state.is_winning())
				result.add(sars)
				find = True
				break
		if find:
			break
	return list(result)

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
	mlp_config = MLPConfig(size, cols)
	mlp = MLP(mlp_config)
	mlp = mlp.float()
	return mlp

def train(device, size, gamma=0.9):
	mlp = make_mlp(size, size-1)
	mlp.to(device)

	replay_buffer = initialize_replay_buffer(load_grids(size), mlp)
	#replay_buffer = initialize_buffer_with_single_test_tuple(size)
	epsilon = 1.0
	optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001) # 
	loss_function = torch.nn.SmoothL1Loss() #torch.nn.MSELoss()

	target_mlp = None
	update_target = 1000
	train_steps = 500000
	average_loss = 0.
	target_state_dict = None
	losses = list()

	update_count = 0

	batch_start = 0

	for i in range(train_steps):
		if epsilon > 0.1 :
			epsilon -=9e-7

		# Sanity check to make sure we're not accidentally modifying the
		# target network in between updates.
		# for key,item in target_state_dict.items():
		# 	if not torch.all(torch.eq(item, target_mlp.state_dict()[key])):
		# 		print("Target has been altered!")

		# Sample a random mini-batch from the replay buffer.
		batch_size = 200 #len(replay_buffer)
		batch = get_mini_batch(replay_buffer, batch_size) #len(replay_buffer))
		batch_end = min(batch_start + batch_size, len(replay_buffer))
		batch_size = batch_end - batch_start
		# print("Batch start: ", batch_start)
		# print("BATCH END: ", batch_end)
		# print("BATCH SIZE: ", batch_size)

		batch = replay_buffer[batch_start:batch_end]
		batch_start += batch_size


		# Update the target nextwork to match the update network.
		#if batch_start >= len(replay_buffer):
		#	batch_start = 0
		update_count += 1

		if update_count == update_target or i == 0:
			print("Update target network!")
			target_mlp = update_target_network(mlp)
			target_state_dict = copy.deepcopy(target_mlp.state_dict())
			update_count = 0
			print(target_mlp)
			pytorch_total_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
			print("TOTAL PARAMS: ", pytorch_total_params)
			# for param in target_mlp.parameters():
			# 	print(param.data)



		target_values = torch.zeros([len(batch), 1], dtype=torch.float32)
		model_values = torch.zeros([len(batch), 1], dtype=torch.float32)


        # zero the parameter gradients
		optimizer.zero_grad()

		for b in range(batch_size):
			# Expand out the tuple.
			state, action, reward, new_state, terminal = batch[b]

			# Get the estimated q values for the current state.
			q_values = mlp(state.get_feature_vector())
			#print("Q VALUES: ", q_values)
			#print("ACTION: ", action)
			q_sa = q_values[action]

			#print(q_values)

			# Get the maximum action for q(s',a'; w-). If we're
			# in the terminal/winning state, then there is no next
			# state, and so q_prime_sa is just 0.
			if not terminal:
				_, q_prime_sa = target_mlp.greedy_action(new_state.get_feature_vector(), grad=False)
			else:
				q_prime_sa = torch.tensor(0.)

			target_values[b] = reward + gamma*q_prime_sa
			model_values[b] = q_sa


			# color = int(action / 4 + 1)
			# direction = int(action % 4)
			# action_tu = (color, direction)
			# print("Action: ", action_tu)
			# print("Action index: ", action)
			# print(q_values)

			# Update the entries in the replay buffer. Don't bother adding winning states in
			# because they have no viable actions.
			if not terminal:
				updated_action, updated_qsa = mlp.get_next_action(new_state.get_feature_vector(), grad=False,exploration_rate=epsilon)
				updated_action_index = (updated_action[0]-1)*4 + updated_action[1]


				updated_state, reward, terminal = new_state.step(updated_action)
				replay_buffer.append((new_state, updated_action_index, reward, updated_state, terminal))
			else:
				print("Winning state!")

		# Get average loss
		loss = loss_function(target_values, model_values)

		# Remove the oldest elements from the replay buffer.
		if len(replay_buffer) > max_items_in_replay:
			replay_buffer = replay_buffer[len(replay_buffer)-max_items_in_replay:]

		# Perform gradient descent.
		loss.backward()
		for param in mlp.parameters():
			param.grad.data.clamp_(-1, 1)
		
		optimizer.step()

		average_loss += loss.item()

		# Print out loss calculations.		
		if i % 100 == 0:
			print("Iteration " + str(i) + " average loss: " + str(average_loss / 100))
			losses.append(average_loss / 100)
			average_loss = 0.0

		if i % 100 == 0:
			torch.save(mlp.state_dict(), "q_models/" + str(size) + "x" + str(size) + "_" + str(i) + "_" + "model_v2.txt")
			

		if i % 50000 == 0 and i != 0:
			plt.plot(losses)
			plt.show()

	return mlp


def play(mlp, size=4, index=1):
	grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_" + str(index) + ".txt")

	# Wrap the states as QStates to get functionality
	# specifically needed for Q-learning.
	state = QState(grid.start_state)
	won = False
	turns = 0
	while turns < 1000:
    	# Grab the feature vector for the given QState.
		features = state.get_feature_vector()

		# Get best action from the MLP.
		action, _ = mlp.get_next_action(features, grad=False, exploration_rate=0.05)
		#print("Take action: ", action)

		turns+=1
		if not state.is_viable_action(action):
			continue

     	# Advance to the next state.
		state = state.next_state(action)

		# Break if we're in the winning state.
		if state.is_winning():
			won = True
			break


	# if won:
	# 	renderer = GridRenderer("Q-Learning")
	# 	renderer.render(state.state)
	# 	renderer.tear_down()

	return won


# Determines the board size we will be using for training.
def get_options():
	parser = OptionParser()

	parser.add_option("-s", "--size",
						action="store", # optional because action defaults to "store"
                      	dest="size",
                      	default=5,
                      	help="Size of board to use",)

	parser.add_option("-m", "--mode",
						action="store", # optional because action defaults to "store"
                      	dest="mode",
                      	default="train",
                      	help="Train or play",)

	parser.add_option("-f", "--file",
					  action="store",
					  dest="file",
					  help="file to load model from")

	return parser.parse_args()

def check_gpu():
	print("Check GPU")
	# cuda_device = torch.cuda.current_device()
	# print(cuda_device)

	# print(torch.cuda.device(0))

	# print("Device count: ", torch.cuda.device_count())

	# print("Device name: ", torch.cuda.get_device_name())

	# print("Device available: ", torch.cuda.is_available())

	if torch.cuda.is_available():
		print("We have cuda!!!")

	return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
	device = check_gpu()

	# Grab the command line options.
	options, args = get_options()
	print("Training with boards of size ", options.size)

	size = int(options.size)
	if options.mode == "train":
		mlp = train(device, size)
	elif options.mode == "play":
		parameters = torch.load("q_models/" + options.file)
		mlp_config = MLPConfig(size, size-1)
		mlp = MLP(mlp_config)
		mlp.load_state_dict(parameters)
		mlp.train(False)

		total_wins = 0
		for i in range(900, 1000):
			wins = 0
			for _ in range(40):
				wins += play(mlp, int(options.size), i)
				#print("We " + ("won \\^_^/" if status else "lost =("))
			print("Board " + str(i) + " won " + str(wins) + " times.")
			if wins > 0:
				total_wins += 1
		print("Total boards won: ", total_wins)

if __name__ == "__main__":
	main()
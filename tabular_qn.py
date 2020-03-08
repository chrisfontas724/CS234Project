from models.grid import Grid
import numpy as np
import random
from optparse import OptionParser
from renderer.renderer import GridRenderer as Renderer
import pickle
import os.path
from os import path

# Train a tabular dqn model here. We can pass in an existing q function to continue training
# from a previous run.
def train(file, size, Q=dict(), gamma=0.9, num_epochs=3):
	print("Train ", file)
	print("Epochs: ", num_epochs)
	grid = Grid(filename=file)
	epsilon = 1.0

	print("Generate all states!")
	all_states = grid.generate_all_states()
	state_size = len(all_states)
	print("All states: ", state_size)

	lr = 0.5
	gamma = 0.9
	winning_states = 0
	action_size = 4 * (size-1) # number of colors is 4 in 5*5 grid

	iter = 0
	for epoch in range(num_epochs):
		print("Epoch: ", epoch)
		for state in all_states:
			for action in range(action_size):

				if iter % 1000 == 0:
					print("iteration ", iter)

				if not state in Q:
					Q[state] = np.zeros((action_size))

				color = int(action /4 + 1)
				direction = int(action % 4)
				action_tu = (color, direction)

				def get_next_tuple():
					if state.is_viable_action(action_tu):
						new_state = state.next_state(action_tu)
						if new_state.is_winning():
							reward = 1000000000
							return new_state, reward
						else:
							flows = new_state.completed_flow_count()
							zeroes = new_state.num_zeroes_remaining()
							reward = -5 * zeroes
							for f in range(flows):
								reward += 1000
							return new_state, reward
					else:
						reward = -1000000
						new_state = state
						return new_state, reward

				new_state, reward = get_next_tuple()

				if not new_state in Q:
					Q[new_state] = np.zeros((action_size))

				Q[state][action] = Q[state][action] + lr * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

				if new_state.is_winning():
					print("Winning State!")
					winning_states += 1
				iter+=1

	print("Number of winning states: ", winning_states)
	return Q


# Play tabular here.
def play(file, Q, size):
	grid = Grid(filename=file)
	print("Playing ", file)

	epsilon = 0.001
	action_size = 4 * (size-1) # number of colors is (size-1), number of directions is 4.

	state = grid.start_state
	won = False

	turns = 0

	while turns < 100000:
		if random.uniform(0, 1) < epsilon or not state in Q:
			action = random.randint(0,action_size-1)
		else:
			action = np.argmax(Q[state])

		color = int(action / 4 + 1)
		direction = int(action % 4)
		action_tu = (color, direction)

		turns += 1

		if not state.is_viable_action(action_tu):
			continue

		# Advance to the next state.
		state = state.next_state(action_tu)

		# Break if we're in the winning state.
		if state.is_winning():
			won = True
			break
	renderer = Renderer("Play")
	renderer.render(state)
	renderer.tear_down()

	print ("Took " + str(turns) + " turns!")
	return won

# Determines the board size we will be using for training.
def get_options():
	parser = OptionParser()

	parser.add_option("-s", "--size",
						action="store", # optional because action defaults to "store"
                      	dest="size",
                      	default=4,
                      	help="Size of board to use",)


	parser.add_option("-m", "--mode",
						action="store", # optional because action defaults to "store"
                      	dest="mode",
                      	default="train",
                      	help="train or test",)


	parser.add_option("-g", "--grid",
					  action="store", # optional because action defaults to "store"
					  default="default",
                      dest="grid",
                      help="grid board number to use",)


	parser.add_option("-e", "--epochs",
					  action="store", # optional because action defaults to "store"
					  default="default",
                      dest="epochs",
                      help="training epochs",)

	return parser.parse_args()


def main():
	# Grab the command line options.
	options, args = get_options()
	grid_num = options.grid
	print("Running with boards of size ", options.size)
	file = "tabular/" + options.size + "x" + options.size + "_grid" + grid_num + ".pickle"
	grid_name = "levels/" + options.size + "x" + options.size + "/" + "grid_" + grid_num + ".txt"
	print("File: ", file)

	if options.mode == "train":
		# If there's already a Q file on disk with this name, load it up
		# again to resume training.
		Q = dict()
		if path.exists(file):
			print("Loading existing train file...")
			with open(file, 'rb') as handle:
				Q = pickle.load(handle)

		# Train Q
		Q = train(file=grid_name, size=int(options.size), Q=Q, gamma=0.9, num_epochs=int(options.epochs))

		# Save Q dictionary back to disk.
		with open(file, 'wb') as handle:
			pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		print("Loading file from disk...")
		with open(file, 'rb') as handle:
			Q = pickle.load(handle)
		final = play(file=grid_name, Q=Q, size=int(options.size))
		print("Results: ", final)

if __name__ == "__main__":
	main()
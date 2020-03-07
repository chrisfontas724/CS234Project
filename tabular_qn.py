from models.grid import Grid
import numpy as np
import random
from optparse import OptionParser
from renderer.renderer import GridRenderer as Renderer






# Train a tabular dqn model here.
def train(size, gamma=0.9):

	grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_" + "1" + ".txt")
	epsilon = 1.0
	#states = grid.generate_all_states()
	#state_size = len(states)
	action_size = 4 * (size-1) # number of colors is 4 in 5*5 grid
	lr = 0.5
	gamma = 0.9

	Q = dict() #np.zeros((state_size, action_size))

	state = grid.start_state
	for i in range(5000000):
		if i%1000==0:
			print("Iteration ", i)
			if epsilon >0.05:
				epsilon -= 0.05


		if not state in Q:
			Q[state] = np.zeros((action_size))


		if random.uniform(0, 1) < epsilon:
			action = random.randint(0,action_size-1)
		else:
			action = np.argmax(Q[state])

		color = int(action / 4 + 1)
		direction = int(action % 4)
		action_tu = (color, direction)


		if state.is_viable_action(action_tu):
			new_state = state.next_state(action_tu)
			reward = 1000 if new_state.is_winning() else 10
		else:
			reward = -10000
			new_state = state


		if not new_state in Q:
			Q[new_state] = np.zeros((action_size))

		Q[state][action] = Q[state][action] + lr * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

		state = new_state if not new_state.is_winning() else grid.start_state

	return Q




# Play tabular here.
def play(Q, size):
	grid = Grid(filename="levels/" + str(size) + "x" + str(size) + "/grid_" + "1" + ".txt")


	state = grid.start_state
	won = False
	while True:

		action = np.argmax(Q[state])

		color = int(action / 4 + 1)
		direction = int(action % 4)
		action_tu = (color, direction)
		print("Take action: ", action_tu)

		if not state.is_viable_action(action_tu):
			break

		# Advance to the next state.
		state = state.next_state(action_tu)

		renderer = Renderer("Play")
		renderer.render(state)
		renderer.tear_down()

		# Break if we're in the winning state.
		if state.is_winning():
			won = True
			break

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

	return parser.parse_args()


def main():
	# Grab the command line options.
	options, args = get_options()
	print("Training with boards of size ", options.size)

	if options.mode == "train":
		Q = train(size=int(options.size), gamma=0.9)
		with open("tabular/" + options.size +"x" + options.size + ".pickle", 'wb') as handle:
			pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		with open("tabular/" + options.size +"x" + options.size + ".pickle", 'rb') as handle:
			Q = pickle.load(handle)
		final = play(Q,size=int(options.size))
		print("Results: ", final)

if __name__ == "__main__":
	main()
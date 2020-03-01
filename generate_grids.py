from models.grid import Grid
from renderer.renderer import GridRenderer
import numpy as np
import random
import os

def filter_actions_by_color(actions, col):
	result = list()
	for action in actions:
		if action[0] == col:
			result.append(action)
	return result


def random_start(size, num_colors):
	spaces = np.zeros((size, size))
	start_coords = dict()
	for color in range(1, num_colors + 1):
		while True:
			row = random.randint(0, size-1)
			col = random.randint(0, size-1)
			if spaces[row][col] == 0:
				spaces[row][col] = color
				start_coords[color] = (row, col)
				break


	return (spaces, start_coords)

def generate_random_grid(size=4, num_colors=3):
	spaces, start_coords = random_start(size, num_colors)

	print("SPACES: ", spaces)
	print("START: ", start_coords)

	grid = Grid.create(spaces, num_colors, start_coords, end_coords=dict())   

	state = grid.start_state
	searching = True
	won = False
	while  searching:
   		for col in range(1, num_colors+1):

   			possible_actions = state.possible_actions(check_end_tips=False)
   			print("POSSIBLE ACTIONS: ", possible_actions)
   			if len(possible_actions) == 0:
   				searching = False
   				break

   			actions_for_color = filter_actions_by_color(possible_actions, col)
   			if len(actions_for_color) == 0:
   				continue

   			random_action = random.choice(actions_for_color)

   			state = state.next_state(random_action, check_end_tips=False)

   			if state.is_winning(check_end_tips=False):
   				print("State won!")
   				searching = False
   				won = True
   				break

	if won:
		state.set_to_start_with_current_tips()
		return state
	else:
		return None


def write_states_to_disk(states, size):
	dir_name = "levels/" + str(size) + "x" + str(size)
	if not os.path.exists(os.path.dirname(dir_name)):
	 	os.mkdir(dir_name)
	counter = 1
	for state in states:
		filename = dir_name + "/" + "grid_" + str(counter) + ".txt"
		#os.mknod(filename)
		file = open(filename, 'a')
		file.truncate(0)
		file.write(str(size) + " " + str(size) + "\n")
		for i in range(size):
			file.write( '    ' + '    '.join(map(str, state.spaces[i].astype(int))) + "\n")
		file.close()
		counter += 1

		


def main():
	size = 5
	num_colors = 4
	states = set()
	while len(states) < 1:
		state = generate_random_grid(size=size, num_colors=num_colors)
		if state is not None:
			states.add(state)

	write_states_to_disk(states, size)
	for state in states:
		renderer = GridRenderer("Generated State")
		renderer.render(state)
		renderer.tear_down()


if __name__ == "__main__":
	main()
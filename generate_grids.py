from models.grid import Grid
from renderer.renderer import GridRenderer
import numpy as np
import random

def filter_actions_by_color(actions, col):
	result = list()
	for action in actions:
		if action[0] == col:
			result.append(action)
	return result


def random_start(size, num_colors):
	pass

def generate_random_grid(size=4, num_colors=3):

	spaces = np.array([[ 1,  2,  3,  0],
						[ 0,  0,  0,  0],
                      	[ 0,  0,  0,  0],
                       	[ 0,  0,  0,  0]])
	start_coords = {
			1:(0,0),
			2:(0,1),
			3:(0,2)
	}

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
		return state
	else:
		return None

def main():
	state = generate_random_grid()
	if state is None:
		print("Failed =(")
		return

	renderer = GridRenderer("Generated State")
	renderer.render(state)
	renderer.tear_down()


if __name__ == "__main__":
	main()
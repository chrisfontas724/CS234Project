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


def generate_random_grid(size=4, num_colors=3):

	spaces = np.array([[ 1,  2,  3,  0],
						[ 0,  0,  0,  0],
                      	[ 0,  0,  0,  0],
                       	[ 1,  2,  3,  0]])
	start_coords = {
			1:(0,0),
			2:(0,1),
			3:(0,2)
	}

	grid = Grid.create(spaces, num_colors, start_coords, end_coords=dict())   

	state = self.start_state
	while True:
   		for col in range(1, num_colors+1):

   			possible_actions = state.possible_actions()
   			if len(possible_actions) == 0:
   				return

   			actions_for_color = filter_actions_by_color(possible_actions, col)
   			random_action = random.choice(actions_for_color)

   			state = state.next_state(random_action)

   			if state.is_winning(check_end_tips=False):
   				return state



def main():
	state = generate_random_grid()

	renderer = GridRenderer("Generated State")
	renderer.render(state)
	renderer.tear_down()


if __name__ == "__main__":
	main()
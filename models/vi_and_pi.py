from models.grid import Grid


# Return a tuple (probability, next_state, reward) for
# a given state/action pair.
def get_tuple(state, action):
	next_state = state.next_state(action)
	reward = 1 if next_state.is_winning() else 0
	return (1, next_state, reward)


# Use this file to fill out policy iteration and value iteration
# methods for solving FlowFree. This is step 1 of our project.

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    pass


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    pass


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    pass


def value_iteration(grid, gamma=0.9, tol=1e-3):
	states = grid.generate_all_states()
	nS = len(states)

	value_function = dict()
	policy = dict()

	while True:
		new_values = value_function.copy()

		delta = 0

		# Iterate over all the states.
		for state in states:

			if not state in new_values:
				new_values[state] = 0.
			old_value = new_values[state]

			max_value = float('-inf')
			max_action = 0

			# Iterate over all actions for this state.
			possible_actions = state.possible_actions()
			for action in possible_actions:
				tuple = get_tuple(state, action)
				new_value = 0.

				# (probability, nextstate, reward)
				(probability, next_state, reward) = tuple
				if not next_state in value_function:
					value_function[next_state] = 0.

				new_value += probability * (reward + gamma * value_function[next_state])

				if new_value > max_value:
					max_value = new_value
					max_action = action

			new_values[state] = max_value
			policy[state] = max_action
			delta = max(delta, max_value - old_value)

		value_function = new_values
		if delta < tol:
			break


	############################
	return value_function, policy
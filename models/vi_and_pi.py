from models.grid import Grid
import numpy as np

# Return a tuple (probability, next_state, reward) for
# a given state/action pair.
def get_tuple(state, action):
	if not state.is_viable_action(action):
		return None
	next_state = state.next_state(action)
	reward = 1 if next_state.is_winning() else 0
	return (next_state, reward)


# Use this file to fill out policy iteration and value iteration
# methods for solving FlowFree. This is step 1 of our project.

def policy_evaluation(states, policy, gamma=0.9, tol=1e-3):


	value_function = dict()
	prev_v_f = value_function.copy()
	while True:

		for state in states:
			v = 0
			if not state in policy:
				policy[state] = (1,0)

			tuple = get_tuple(state,policy[state])
			if tuple is not None:
				next_state, reward = tuple
				if not next_state in prev_v_f:
					prev_v_f[next_state]  = 0

				v +=  (reward + gamma * prev_v_f[next_state])
				value_function[state] = v
		max = 0
		for key, values in value_function.items():
			if key not in prev_v_f:
				prev_v_f[key] = 0
			tol_current = value_function[key]-prev_v_f[key]
			if tol_current>max:
				max = tol_current

		if tol_current < tol:
			break
		prev_v_f = value_function.copy()
	return value_function


def policy_improvement(states, value_from_policy, gamma=0.9):

	new_policy = dict()
	for state in states:

		possible_actions = state.possible_actions()
		Q_pi = dict()
		for action in possible_actions:
			Q_pi[action] = 0.

		for action in possible_actions:

			tuple = get_tuple(state, action)
			if tuple is not None:
				next_state, reward = tuple
				if next_state not in value_from_policy:
					value_from_policy[next_state] = 0
				Q_pi[action] +=  reward + gamma * value_from_policy[next_state]

		best_action = (1,0)
		best_val = 0
		for key, val in Q_pi.items():
			if val > best_val:
				best_action = key
				best_val = val
		new_policy[state] = best_action

		return new_policy


def policy_iteration(grid,gamma=0.9, tol=10e-3):

	states = grid.generate_all_states()
	policy = dict()
	#value_from_policy = dict()
	#policy_new = dict()
	while True:
		value_from_policy = policy_evaluation(states, policy, gamma, tol)
		policy_new = policy_improvement(states, value_from_policy, gamma)
		if policy==policy_new:
			policy = policy_new
			break
		policy = policy_new.copy()

	return policy


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
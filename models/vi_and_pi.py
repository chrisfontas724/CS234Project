from models.grid import Grid
import numpy as np

# Return a tuple (next_state, reward) for
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

	while True:
		old_values = value_function.copy()

		# Loop over all states
		for state in states:
			if not state in policy:
				policy[state] = (1,0)

			if state.is_viable_action(policy[state]):
				next_state, reward = get_tuple(state, policy[state]) 

				total_value = 0.
				old_reward = 0 if not next_state in old_values else old_values[next_state]
				total_value = (reward + gamma * old_reward)
				value_function[state] = total_value

		max = 0
		for key, values in value_function.items():
			if key not in old_values:
				old_values[key] = 0
			tol_current = value_function[key]-old_values[key]
			if tol_current>max:
				max = tol_current

		if max < tol:
			break

	############################
	return value_function

def policy_improvement(states, value_from_policy, gamma=0.9):
	new_policy = dict()

	# Loop over all the states.
	for state in states:
		max_val = 0.0

		# Get the action that maximizes the policy for the current state.
		for action in state.possible_actions():
			(next_state, reward) = get_tuple(state, action)
			curr_val = reward
			if next_state in value_from_policy:
				curr_val += gamma * value_from_policy[next_state]
			if curr_val >= max_val:
				max_val = curr_val
				new_policy[state] = action

	############################
	return new_policy

def policy_iteration(grid,gamma=0.9, tol=10e-3):
	print("Begin policy iteration...")
	states = grid.generate_all_states()
	policy = dict()

	value_function = dict()
	policy = dict()

	while True:

		new_policy = policy_improvement(states, value_function, gamma)

		new_values = policy_evaluation(states, new_policy, gamma, tol)

		# Find the maximum difference between old and new values.
		policy_changed = 0.
		for key,value in new_values.items():
			if not key in value_function:
				value_function[key] = 0.
			change = np.abs(value - value_function[key])
			if change > policy_changed:
				policy_changed = change

		# Break if policy below tolerance.
		if policy_changed < tol:
			break
		
		# Assign new policy
		policy = new_policy.copy()
		value_function = new_values.copy()

	return value_function, policy

def value_iteration(grid, gamma=0.9, tol=1e-3):
	print("Begin value iteration...")
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

				# (nextstate, reward)
				(next_state, reward) = tuple
				if not next_state in value_function:
					value_function[next_state] = 0.

				new_value += (reward + gamma * value_function[next_state])

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
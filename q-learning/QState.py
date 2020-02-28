from models.grid import Grid
import numpy as np


# Extension of the state class used in dynamic programming that has
# added functionality to make it easier to use during Q Learning.
class QState(Grid.State):
	def __init__(self, state):
        super(Grid.State, self).__init__()

    # Return a feature vector for the given state to pass into. The
    # feature vector is a flat array consisting of the flattened
    # state of the board as well as the current flow tips appened
    # to the end.
    def get_feature_vector(self):
    	flattened_spaces = self.spaces.flatten()
    	for col in range(1, self.info.num_cols + 1):
    		tip = self.tips[col];
    		flattened_spaces.append(tip[0])
    		flattened_spaces.append(tip[1])
    	return flattened_spaces

    # This overrides the |next state| function from the parent class.
    # The difference here is that since we are doing Q-learning, actions
    # that are impossible may be passed to the current state, in which case
    # we don't want to simply abort, but we should wind up in the same
    # state again.
    #
    # We also want to return rewards here too, alongside the next state. If
    # an action fails and we wind up in the same state twice, we should return
    # a very negative reward. All other states get a minor negative reward,
    # except for the winning state which gets a large positive reward.
    def next_state(self, action):

    	# We need to penalize impossible actions very highly.
    	if not self.is_viable_action(action):
    		return self.copy(), -1000000000
    	else:
    		state = super().next_state(action) if self.is_viable_action(action) else self.copy()
    		reward = 100 if state.is_winning() else -1
    		return state, reward


if __name__ == "__main__":
	main()
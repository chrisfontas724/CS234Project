from models.grid import Grid
import numpy as np
import torch


# Extension of the state class used in dynamic programming that has
# added functionality to make it easier to use during Q Learning.
class QState(Grid.State):
    def __init__(self, state):
        self.state = state

    # Return a feature vector for the given state to pass into. The
    # feature vector is a flat array consisting of the flattened
    # state of the board as well as the current flow tips appened
    # to the end.
    def get_feature_vector(self):
        flattened_spaces = self.state.spaces.flatten()
        for col in range(1, self.state.info.num_cols + 1):
            tip = self.state.tips[col]
            flattened_spaces = np.append(flattened_spaces, tip[0])
            flattened_spaces = np.append(flattened_spaces, tip[1])
        return torch.from_numpy(flattened_spaces)


    def is_winning(self):
        return self.state.is_winning()

    def is_viable_action(self, action):
        return self.state.is_viable_action(action)

    def next_state(self, action_tuple):
        return QState(self.state.next_state(action_tuple))

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
    #
    # Lastly, we also return if we are done or not (based on whether or not
    # we've reached a winning state). This is used by the Q-learning algorithm
    # to determine if it should break or continue.
    def step(self, action):

    	# We need to penalize impossible actions very highly.
    	if not self.state.is_viable_action(action):
    		return (self, -100, False)
    	else:
    		state = self.next_state(action)
    		won = state.is_winning()
    		reward = 100 if won else -1
    		return (state, reward, won)


if __name__ == "__main__":
	main()
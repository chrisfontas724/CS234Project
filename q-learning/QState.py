from models.grid import Grid
import numpy as np


# Extension of the state class used in dynamic programming that has
# added functionality to make it easier to use during Q Learning.
class QState(Grid.State):
	def __init__(self, info, spaces, tips):
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


if __name__ == "__main__":
	main()
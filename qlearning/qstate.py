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
    # state of the board as well as the current flow tips append
    # to the end.
    def get_feature_vector(self):

        # Start the feature vector off with the flattened positions
        # of all the flows.
        flattened_spaces = self.state.spaces.flatten()

        # Add the starting points, ending points, and current tips
        # to the feature vector
        start_points = self.state.info.color_start_coords
        end_points = self.state.info.color_end_coords
        for col in range(1, self.state.info.num_cols + 1):
            tip = self.state.tips[col]
            start = start_points[col]
            end = end_points[col]
            flattened_spaces = np.append(flattened_spaces, tip[0])
            flattened_spaces = np.append(flattened_spaces, tip[1])
            flattened_spaces = np.append(flattened_spaces, start[0])
            flattened_spaces = np.append(flattened_spaces, start[1])
            flattened_spaces = np.append(flattened_spaces, end[0])
            flattened_spaces = np.append(flattened_spaces, end[1])

        # Add in the number of possible actions.
        flattened_spaces = np.append(flattened_spaces, len(self.state.possible_actions()))

        # Add in if this is a winning state or not.
        flattened_spaces = np.append(flattened_spaces, int(self.is_winning()))


        # Make sure the vector is in float format before returning.
        return torch.from_numpy(flattened_spaces).float()

    def __hash__(self):
        return hash((str(self.state.spaces.tolist()), str(self.state.tips)))

    def is_winning(self):
        return self.state.is_winning()

    def is_viable_action(self, action):
        return self.state.is_viable_action(action)

    def next_state(self, action_tuple):
        return QState(self.state.next_state(action_tuple))

    def num_zeroes_remaining(self):
        return self.state.num_zeroes_remaining()

    def completed_flow_count(self):
        return self.state.completed_flow_count()

    # This overrides the |next state| function from the parent class.
    # The difference here is that since we are doing Q-learni
    # ng, actions
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
            return (self, -1000, False) #returns new state as current state
        else:
            state = self.next_state(action)
            won = state.is_winning()

            if won is True:
                reward = 10000000
                return(state, reward, won)

            else:

                zeros = self.state.num_zeroes_remaining() - state.num_zeroes_remaining()
                reward = 100 * zeros


                flow  = state.completed_flow_count() -  self.state.completed_flow_count()
                reward += 100 * flow

                return (state, reward, won)





    #number of flows, number of zeros


if __name__ == "__main__":
	main()
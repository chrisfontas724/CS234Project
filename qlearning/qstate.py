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

        # Add in the number of zeroes
        flattened_spaces = np.append(flattened_spaces, self.num_zeroes_remaining())

        # Add in the number of flows.
        flattened_spaces = np.append(flattened_spaces, self.completed_flow_count())




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
            return (self, -10.0, False) #returns new state as current state
        else:
            state = self.next_state(action)
            won = state.is_winning()

            if won is True:
                reward = 100.0
                return(state, reward, True)

            else:

                # If the new state has fewer zeros than the old state, this should be rewarded.
                # Otherwise we should subtract points if the new state has more zeroes.
                zeros = state.num_zeroes_remaining() #  self.state.num_zeroes_remaining() - state.num_zeroes_remaining()
               # reward = -1.5 * zeros
                reward = 0

                # # If the new state has more flows than the old state, add points. If it has fewer
                # # flows, that mean a flow was broken and so we should remove points.
                flow  = state.completed_flow_count() -  self.state.completed_flow_count()
                if flow > 0:
                    reward += flow
                else:
                    reward +=  2 * flow

                return (state, reward, False)





    #number of flows, number of zeros


if __name__ == "__main__":
	main()
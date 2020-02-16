import numpy as np
from optparse import OptionParser
import itertools


# This class represents the board to be used in flow free. It has square dimensions N and
# contains a number of pairs of colored dots located at unique locations (i,j).
# TODO: Fill this out more.
class Grid:
    def __init__(self, filename):
        print("Initializing grid....")

        file = open(filename, "r") 
        counter = 0
        while True:
            line = file.readline()
            if not line:
                break

            # Grab the dimensions of the grid from the
            # first line.
            if counter == 0:
                # Read the line, remove unneeded characters, convert to list then convert
                # each item in the list to an integer.
                dimensions = [int(numeric_string) for numeric_string in line.replace("\n", "").split(' ')] 

                # We should have two entries, otherwise the file format is incorrect.
                assert len(dimensions) == 2

                # Initialize the size and the 2D numpy
                # array that will represent the board.
                self.size = dimensions[0]
                self.spaces = np.zeros((self.size, self.size))

            # Else populate the current row we are on from the data in the line.
            else:
                # Kind of hacky...but basically just removes the invalid characters I found and converts
                # them to integers if they're valid numbers.
                row = [int(numeric_string) for numeric_string in line.replace("\n", "").replace('\t', ' ').split(' ') if numeric_string is not ''] 

                # Row length should match the size, otherwise the file format is wrong.
                assert len(row) == self.size, \
                    "length is %r but should be %r" % (len(row), self.size)

                # Assign row to numpy array.
                self.spaces[counter-1] = row

            counter += 1

            self.num_cols = 3 # HACK


        file.close()

        # Declare dictionaries to store the starting and ending positions for the
        # "flows" for each color. For simplicity, and due to the fact that it doesn't
        # affect the algorithm at all, we assume that all flows must start from one
        # color and make its way to the other, instead of allowing the flow to start
        # from either color arbitrarily.
        self.color_start_coords = dict()
        self.color_end_coords = dict()
        for row in range(self.size):
            for col in range(self.size):
                item = int(self.spaces[row][col])
                if item != 0:
                    if item in self.color_start_coords:
                        assert item not in self.color_end_coords
                        self.color_end_coords[item] = (row, col)
                    else:
                        self.color_start_coords[item] = (row, col)

    # Given an initial board configuration, generate and return a
    # vector of all possible grid configurations. The total number
    # for a 4x4 grid with 3 colors should be roughly around 1M.
    # It should be possible to tighten the number of possible states
    # by disallowing certain impossible configurations, but for now
    # we are defining states more loosely for simplicity.
    def generate_all_states(self, initial_state):
        ranges = list()
        for x in range(self.size):
            for y in range(self.size):
                # If the initial state already has a value at a coordinate (x,y) then
                # that is a fixed point that should not be altered. This is represented
                # by setting its range to (val, val+1) so that its always set to |val|.
                # Otherwise, the range should be [0, num_cols + 1).
                val = int(initial_state[x][y])
                ranges.append(range(val,val+1) if val != 0 else range(0, self.num_cols+1))

        # itertools.product is just a compact way of doing a series of nested iterations.
        # Each nested iteration uses the range provided at its index. We use a flat
        # representation of the grid here because that's what itertools.product() takes
        # in as an argument. We unflatten it later with np.reshape().
        flat_result = list(itertools.product(*ranges, repeat=1))

        # The number of total possible states should be equal to (num_cols+1) ^ (num_empty_spaces).
        # The base is num_cols + 1 to account for empty spaces. So if your colors are [1,2,3], the
        # possible values for a non-empty space are [0,1,2,3].
        assert len(flat_result) == np.power(self.num_cols + 1, (self.size * self.size) - 2*self.num_cols)

        # Now we have to reshape the result so it can be a (sizexsize) grid.
        result = [np.reshape(x, (-1, self.size)) for x in flat_result]
        return result


    # Given an input state, return a list of possible actions that
    # can be taken from this state. At most, the number of moves is
    # (num_colors * num_directions).
    def possible_actions(self, state):
        pass


    # Given a state paired with a particular action, return the next
    # state that would be resulted in. Actions are defined as tuples
    # consisting of a color to move, and the direction in which to
    # move it. For example (red, up) and (blue, left). These values
    # are represented numerically, however, so (red, up) would be
    # (1,0) where "1" represents the color red and "0" represents
    # the direction up.
    def next_state(self, state, action):

        color, direction = action

        result = state.copy()



        pass

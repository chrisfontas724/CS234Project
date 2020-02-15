import numpy as np
from optparse import OptionParser


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
                assert len(row) == self.size

                # Assign row to numpy array.
                self.spaces[counter-1] = row

            counter += 1
        file.close()

    # Given an initial board configuration, generate and return a
    # vector of all possible grid configurations. The total number
    # for a 4x4 grid with 3 colors should be roughly around 1M.
    def generate_all_states(initial_state):
        pass


    # Given an input state, return a list of possible actions that
    # can be taken from this state. At most, the number of moves is
    # (num_colors * num_directions).
    def possible_actions(state):
        pass


    # Given a state paired with a particular action, return the next
    # state that would be resulted in.
    def next_state(state, action)
        pass

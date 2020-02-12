import numpy as np
from optparse import OptionParser

color_dictionary = {
  # TODO - Fill this out
}

# This class represents the board to be used
# in flow free. It has square dimensions N and
# contains a number of pairs of colored dots
# located at unique locations (i,j)
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
                dimensions = line.split(' ')
                assert len(dimensions) == 2
                self.size = dimensions[0]
                print("Grid size: ", str(self.size))
            # Else populate the grid from the data directly.
            else:
                row = line.split(' ')
            counter += 1
        file.close()
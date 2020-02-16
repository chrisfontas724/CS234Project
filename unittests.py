import numpy as np
from optparse import OptionParser
from models.grid import Grid
import unittest

class TestGridFunctions(unittest.TestCase):

    def test_starting_moves(self):
        grid = Grid(filename="levels/test_level.txt")
        self.assertEqual(len(grid.possible_actions(grid.spaces)), 4)


# Program entry point.
if __name__ == "__main__":
	unittest.main()

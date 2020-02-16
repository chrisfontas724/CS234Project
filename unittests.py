import numpy as np
from optparse import OptionParser
from models.grid import Grid
import unittest

class TestGridFunctions(unittest.TestCase):

    def test_basic_grid(self):
        # Create grid.
        grid = Grid(filename="levels/test_level.txt")

        # Make sure the start flow locations are accurate.
        self.assertEqual(grid.color_start_coords[1], (0,0))
        self.assertEqual(grid.color_start_coords[2], (0,1))
        self.assertEqual(grid.color_start_coords[3], (0,2))

        # Make sure the end flow locations are accurate.
        self.assertEqual(grid.color_end_coords[1], (3,0))
        self.assertEqual(grid.color_end_coords[2], (3,1))
        self.assertEqual(grid.color_end_coords[3], (3,2))

        # There should only be 4 moves available at the beginning.
        self.assertEqual(len(grid.possible_actions(grid.spaces)), 4)

        # Move red down, now there should be 5 moves available.
        next_state = grid.next_state((1,2))
        self.assertEqual(len(grid.possible_actions(next_state)), 5)



# Program entry point.
if __name__ == "__main__":
	unittest.main()

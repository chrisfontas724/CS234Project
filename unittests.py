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

        # The following bad moves should throw an exception due to
        # the assert inside |possible_actions|.
        with self.assertRaises(Exception) as context:
            grid.next_state(grid.spaces, (1,0))
        self.assertTrue('Action is not viable' in str(context.exception))

        # Move red down, now there should be 5 moves available and the
        # current tip for red should be at (1,0)
        next_state = grid.next_state(grid.spaces, (1,2))
        self.assertEqual(len(grid.possible_actions(next_state)), 5)
        self.assertEqual(grid.color_flow_tips[1], (1,0))

        # Now try to move red back up, which it shouldn't be able to
        # do because that's where it just came from.
        with self.assertRaises(Exception) as context:
            grid.next_state(next_state, (1,0))
        self.assertTrue('Action is not viable' in str(context.exception))





# Program entry point.
if __name__ == "__main__":
	unittest.main()

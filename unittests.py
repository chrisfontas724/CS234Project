import numpy as np
from optparse import OptionParser
from models.grid import Grid
import unittest

class TestGridFunctions(unittest.TestCase):

    # Test to make sure that the static Grid function |is_valid_state| works
    # as intended.
    def test_valid_states(self):
        grid = np.array([[ 1,  2,  3,  0],
                         [ 0,  0,  0,  0],
                         [ 0,  0,  0,  0],
                         [ 1,  2,  3,  0]])

        start_coords = {
            1:(0,0),
            2:(0,1),
            3:(0,2)
        }
        end_coords = {
            1:(3,0),
            2:(3,1),
            3:(3,2)
        }

        def print_grid(valid):
            print("Valid" if valid else "NOT valid")
            print(grid)
            print("\n")

        # Default starter grid should work just fine.
        print_grid(True)
        self.assertTrue(Grid.is_valid_state(grid, 4, start_coords, end_coords))

        # Add a color to the middle of the grid that is not connected.
        grid[2][2] = 2
        print_grid(False)
        self.assertFalse(Grid.is_valid_state(grid, 4, start_coords, end_coords))

        # Now add another 2 that connects with the above 2, but neither connects
        # with the starting 2.
        grid[1][2] = 2
        print_grid(False)
        self.assertFalse(Grid.is_valid_state(grid, 4, start_coords, end_coords))

        # Now add another 2 so that all the 2s connect to the starting 2.
        grid[1][1] = 2
        print_grid(True)
        self.assertFalse(Grid.is_valid_state(grid, 4, start_coords, end_coords))

        # Remove the above 2, turn the other 2 into a 3 and it should be invalid once more.
        grid[1][1] = 0
        grid[1][2] = 3
        print_grid(False)
        self.assertFalse(Grid.is_valid_state(grid, 4, start_coords, end_coords))

        # Turn the other 2 into a 3, and now we should have 3 connected 3s.
        grid[2][2] = 3
        print_grid(True)
        self.assertTrue(Grid.is_valid_state(grid, 4, start_coords, end_coords))

        # Add a 3 in the corner. This should be false since we can't have two
        # connections with the starting or ending colors.
        grid[0][3] = 3
        print_grid(False)
        self.assertFalse(Grid.is_valid_state(grid, 4, start_coords, end_coords))

        # Now make a grid of 3s around the starting 3 - This should be valid.
        grid[2][2] = 0
        grid[1][3] = 3
        print_grid(True)
        self.assertTrue(Grid.is_valid_state(grid, 4, start_coords, end_coords))

        # Add an extra 3 so the 3s snake around a bit, this should still be valid.
        grid[2][2] = 0
        grid[1][3] = 3
        grid[2][2] = 3
        print_grid(True)
        self.assertTrue(Grid.is_valid_state(grid, 4, start_coords, end_coords))

        # Make sure that the winning state is valid.
        grid[1,0] = 1
        grid[2,0] = 1
        grid[1,1] = 2
        grid[2,1] = 2
        grid[2,2] = 3
        grid[3,2] = 3
        grid[2,3] = 3
        grid[3,3] = 3
        print_grid(True)
        self.assertTrue(Grid.is_valid_state(grid, 4, start_coords, end_coords))


    # Test to make sure we are calculating flow tips correctly.
    def test_flow_tips(self):
        grid = np.array([[ 1,  2,  3,  0],
                         [ 0,  0,  0,  0],
                         [ 0,  0,  0,  0],
                         [ 1,  2,  3,  0]])

        start_coords = {
            1:(0,0),
            2:(0,1),
            3:(0,2)
        }
        end_coords = {
            1:(3,0),
            2:(3,1),
            3:(3,2)
        }

        # The flow tips for the first 3 colors should be the starting coordinates.
        self.assertEqual(Grid.get_flow_tip(grid, 4, 1, start_coords), start_coords[1])
        self.assertEqual(Grid.get_flow_tip(grid, 4, 2, start_coords), start_coords[2])
        self.assertEqual(Grid.get_flow_tip(grid, 4, 3, start_coords), start_coords[3])

        # Move the red (1) flow down and over.
        grid[1,0] = 1
        grid[1,1] = 1
        self.assertEqual(Grid.get_flow_tip(grid, 4, 1, start_coords), (1,1))

        # Move the 2 flows all the way down to the end.
        grid[1][1] = 2
        grid[2][1] = 2
        self.assertEqual(Grid.get_flow_tip(grid, 4, 2, start_coords), end_coords[2])

        # Make a grid with the 3 flows.
        grid[0][3] = 3
        grid[1][2] = 3
        grid[1][3] = 3
        self.assertEqual(Grid.get_flow_tip(grid, 4, 3, start_coords), (1,2))

        # Make the other 2 a 3.
        grid[1,1] = 3
        print(grid)
        self.assertEqual(Grid.get_flow_tip(grid, 4, 3, start_coords), (1,1))


    def test_basic_grid(self):
        # Create grid.
        grid = Grid(filename="levels/test_level.txt")

        print("Number of valid states: ", len(grid.possible_states))

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

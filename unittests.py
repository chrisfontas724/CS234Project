import numpy as np
from optparse import OptionParser
from models.grid import Grid
import unittest

class TestGridFunctions(unittest.TestCase):

    # Test to make sure that the static Grid function |is_valid_state| works
    # as intended.
    def test_valid_states(self):
        spaces = np.array([[ 1,  2,  3,  0],
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

        grid = Grid.create(spaces, 3, start_coords, end_coords)

        def print_grid(valid, state):
            print("Valid" if valid else "NOT valid")
            print(state.spaces)
            print("\n")

        # Default starter grid should work just fine.
        state = grid.start_state
        print_grid(True, state)
        self.assertTrue(state.is_valid())

        # Add a color to the middle of the grid that is not connected.
        state.spaces[2][2] = 2
        print_grid(False, state)
        self.assertFalse(state.is_valid())

        # Now add another 2 that connects with the above 2, but neither connects
        # with the starting 2.
        state.spaces[1][2] = 2
        print_grid(False, state)
        self.assertFalse(state.is_valid())

        # Now add another 2 so that all the 2s connect to the starting 2.
        state.spaces[1][1] = 2
        print_grid(True, state)
        self.assertFalse(state.is_valid())

        # Remove the above 2, turn the other 2 into a 3 and it should be invalid once more.
        state.spaces[1][1] = 0
        state.spaces[1][2] = 3
        print_grid(False, state)
        self.assertFalse(state.is_valid())

        # Turn the other 2 into a 3, and now we should have 3 connected 3s.
        state.spaces[2][2] = 3
        print_grid(True, state)
        self.assertTrue(state.is_valid())

        # Add a 3 in the corner. This should be false since we can't have two
        # connections with the starting or ending colors.
        state.spaces[0][3] = 3
        print_grid(False, state)
        self.assertFalse(state.is_valid())

        # Now make a grid of 3s around the starting 3 - This should be valid.
        state.spaces[2][2] = 0
        state.spaces[1][3] = 3
        print_grid(True, state)
        self.assertTrue(state.is_valid())

        # Add an extra 3 so the 3s snake around a bit, this should still be valid.
        state.spaces[2][2] = 0
        state.spaces[1][3] = 3
        state.spaces[2][2] = 3
        print_grid(True, state)
        self.assertTrue(state.is_valid())

        # Make sure that the winning state is valid.
        state.spaces[1,0] = 1
        state.spaces[2,0] = 1
        state.spaces[1,1] = 2
        state.spaces[2,1] = 2
        state.spaces[2,2] = 3
        state.spaces[3,2] = 3
        state.spaces[2,3] = 3
        state.spaces[3,3] = 3
        print_grid(True, state)
        self.assertTrue(state.is_valid())


    # # Test to make sure we are calculating flow tips correctly.
    # def test_flow_tips(self):
    #     return
    #     grid = np.array([[ 1,  2,  3,  0],
    #                      [ 0,  0,  0,  0],
    #                      [ 0,  0,  0,  0],
    #                      [ 1,  2,  3,  0]])

    #     start_coords = {
    #         1:(0,0),
    #         2:(0,1),
    #         3:(0,2)
    #     }
    #     end_coords = {
    #         1:(3,0),
    #         2:(3,1),
    #         3:(3,2)
    #     }

    #     # The flow tips for the first 3 colors should be the starting coordinates.
    #     self.assertEqual(Grid.get_flow_tip(grid, 4, 1, start_coords, end_coords), start_coords[1])
    #     self.assertEqual(Grid.get_flow_tip(grid, 4, 2, start_coords, end_coords), start_coords[2])
    #     self.assertEqual(Grid.get_flow_tip(grid, 4, 3, start_coords, end_coords), start_coords[3])

    #     # Move the red (1) flow down and over.
    #     grid[1,0] = 1
    #     grid[1,1] = 1
    #     self.assertEqual(Grid.get_flow_tip(grid, 4, 1, start_coords, end_coords), (1,1))

    #     # Move the 2 flows all the way down to the end.
    #     grid[1][1] = 2
    #     grid[2][1] = 2
    #     self.assertEqual(Grid.get_flow_tip(grid, 4, 2, start_coords, end_coords), end_coords[2])

    #     # Make a grid with the 3 flows.
    #     grid[0][3] = 3
    #     grid[1][2] = 3
    #     grid[1][3] = 3
    #     self.assertEqual(Grid.get_flow_tip(grid, 4, 3, start_coords, end_coords), (1,2))

    #     # Make the other 2 a 3.
    #     grid[1,1] = 3
    #     print(grid)
    #     self.assertEqual(Grid.get_flow_tip(grid, 4, 3, start_coords, end_coords), (1,1))


    #     state = np.array([[ 1,  2,  3,  3],
    #                       [ 1,  2,  3,  3],
    #                       [ 1,  2,  3,  3],
    #                       [ 1,  2,  3,  3]])
    #     self.assertEqual(Grid.get_flow_tip(state, 4, 1, start_coords, end_coords), end_coords[1])
    #     self.assertEqual(Grid.get_flow_tip(state, 4, 2, start_coords, end_coords), end_coords[2])
    #     self.assertEqual(Grid.get_flow_tip(state, 4, 3, start_coords, end_coords), end_coords[3])


    # # Do some more advanced flow tip testing on a 5x5 grid with a single color.
    # def test_flow_tips_part_2(self):
    #     return
    #     state = np.array([[ 1,  0,  0,  0, 0],
    #                       [ 0,  0,  0,  0, 0],
    #                       [ 0,  0,  1,  0, 0],
    #                       [ 0,  0,  0,  0, 0],
    #                       [ 0,  0,  0,  0, 0]])

    #     start_coords = {
    #         1:(2,2),
    #     }
    #     end_coords = {
    #         1:(0,0),
    #     }

    #     self.assertEqual(Grid.get_flow_tip(state, 5, 1, start_coords, end_coords), start_coords[1])


    #     state = np.array([[ 1,  0,  0,  0, 0],
    #                       [ 0,  0,  0,  0, 0],
    #                       [ 0,  1,  1,  0, 0],
    #                       [ 0,  1,  1,  0, 0],
    #                       [ 0,  0,  0,  0, 0]])
    #     self.assertEqual(Grid.get_flow_tip(state, 5, 1, start_coords, end_coords), (2,1))


    #     state = np.array([[ 1,  0,  0,  0, 0],
    #                       [ 0,  0,  0,  0, 0],
    #                       [ 0,  1,  1,  0, 0],
    #                       [ 0,  1,  1,  1, 0],
    #                       [ 0,  0,  0,  0, 0]])
    #     self.assertTrue(Grid.is_valid_state(state, 5, start_coords, end_coords))
    #     self.assertEqual(Grid.get_flow_tip(state, 5, 1, start_coords, end_coords), (3,3))

    #     state = np.array([[ 1,  0,  0,  0, 0],
    #                       [ 0,  1,  0,  0, 0],
    #                       [ 0,  1,  1,  0, 0],
    #                       [ 0,  1,  1,  0, 0],
    #                       [ 0,  0,  0,  0, 0]])
    #     self.assertTrue(Grid.is_valid_state(state, 5, start_coords, end_coords))
    #     self.assertEqual(Grid.get_flow_tip(state, 5, 1, start_coords, end_coords), (1,1))

    #     state = np.array([[ 1,  0,  0,  0, 0],
    #                       [ 0,  0,  0,  0, 0],
    #                       [ 0,  0,  1,  1, 0],
    #                       [ 0,  1,  1,  1, 0],
    #                       [ 0,  1,  0,  0, 0]])
    #     self.assertEqual(Grid.get_flow_tip(state, 5, 1, start_coords, end_coords), (4,1))

    #     # Really windy snake example.
    #     state = np.array([[ 1,  0,  0,  0, 1],
    #                       [ 0,  0,  0,  0, 1],
    #                       [ 0,  0,  1,  1, 1],
    #                       [ 0,  1,  1,  1, 1],
    #                       [ 0,  1,  1,  1, 1]])
    #     self.assertEqual(Grid.get_flow_tip(state, 5, 1, start_coords, end_coords), (0,4))

    #     # What if we wind another way?
    #     state = np.array([[ 1,  0,  0,  0, 0],
    #                       [ 0,  0,  1,  1, 0],
    #                       [ 1,  1,  1,  1, 0],
    #                       [ 0,  1,  1,  1, 0],
    #                       [ 0,  0,  0,  0, 0]])
    #     self.assertEqual(Grid.get_flow_tip(state, 5, 1, start_coords, end_coords), (2,0))


    #     state = np.array([[ 1,  0,  0,  0, 0],
    #                       [ 1,  0,  0,  0, 0],
    #                       [ 1,  1,  1,  0, 0],
    #                       [ 0,  0,  0,  0, 0],
    #                       [ 0,  0,  0,  0, 0]])
    #     self.assertEqual(Grid.get_flow_tip(state, 5, 1, start_coords, end_coords), (0,0))


    #     # state = np.array([[ 1,  0,  0,  0, 0],
    #     #                   [ 1,  1,  0,  0, 0],
    #     #                   [ 1,  1,  1,  0, 0],
    #     #                   [ 0,  0,  0,  0, 0],
    #     #                   [ 0,  0,  0,  0, 0]])
    #     # self.assertEqual(Grid.get_flow_tip(state, 5, 1, start_coords, end_coords), (1,1))

    #     # This is a trick...should NOT be valid.
    #     state = np.array([[ 1,  0,  0,  0, 0],
    #                       [ 0,  0,  0,  0, 0],
    #                       [ 0,  1,  1,  0, 0],
    #                       [ 0,  1,  1,  0, 0],
    #                       [ 0,  1,  0,  0, 0]])
    #     self.assertFalse(Grid.is_valid_state(state, 5, start_coords, end_coords))

    # # Test to see if we can determine a winning board or not.
    # def test_winning_board(self):
    #     return
    #     state = np.array([[ 1,  2,  3,  0],
    #                      [ 0,  0,  0,  0],
    #                      [ 0,  0,  0,  0],
    #                      [ 1,  2,  3,  0]])

    #     start_coords = {
    #         1:(0,0),
    #         2:(0,1),
    #         3:(0,2)
    #     }
    #     end_coords = {
    #         1:(3,0),
    #         2:(3,1),
    #         3:(3,2)
    #     }

    #     # Starting state is NOT the winning state.
    #     self.assertFalse(Grid.in_winning_state(state, 4, 3, start_coords, end_coords))

    #     # Flows are all connected but there are still 0s.
    #     state = np.array([[ 1,  2,  3,  0],
    #                       [ 1,  2,  3,  0],
    #                       [ 1,  2,  3,  0],
    #                       [ 1,  2,  3,  0]])
    #     self.assertFalse(Grid.in_winning_state(state, 4, 3, start_coords, end_coords))

    #     # Now we win!
    #     state = np.array([[ 1,  2,  3,  3],
    #                       [ 1,  2,  3,  3],
    #                       [ 1,  2,  3,  3],
    #                       [ 1,  2,  3,  3]])
    #     self.assertTrue(Grid.in_winning_state(state, 4, 3, start_coords, end_coords))

    #     # This other configuration should also win.
    #     state = np.array([[ 1,  2,  3,  3],
    #                       [ 1,  2,  2,  3],
    #                       [ 1,  2,  2,  3],
    #                       [ 1,  2,  3,  3]])
    #     self.assertTrue(Grid.in_winning_state(state, 4, 3, start_coords, end_coords))

    # def test_basic_grid(self):
    #     # Create grid.
    #     grid = Grid(filename="levels/test_level.txt")

    #     # Originally this is 5241
    #     print("Number of valid states: ", len(grid.possible_states))

    #     print("Number of valid states part 2: ", grid.generate_all_states_test(grid.spaces, grid.color_start_coords))

    #     return

    #     # Make sure the start flow locations are accurate.
    #     self.assertEqual(grid.color_start_coords[1], (0,0))
    #     self.assertEqual(grid.color_start_coords[2], (0,1))
    #     self.assertEqual(grid.color_start_coords[3], (0,2))

    #     # Make sure the end flow locations are accurate.
    #     self.assertEqual(grid.color_end_coords[1], (3,0))
    #     self.assertEqual(grid.color_end_coords[2], (3,1))
    #     self.assertEqual(grid.color_end_coords[3], (3,2))

    #     # There should only be 4 moves available at the beginning.
    #     self.assertEqual(len(grid.possible_actions(grid.spaces)), 4)

    #     # The following bad moves should throw an exception due to
    #     # the assert inside |possible_actions|.
    #     with self.assertRaises(Exception) as context:
    #         grid.next_state(grid.spaces, (1,0))
    #     self.assertTrue('Action is not viable' in str(context.exception))

    #     # Move red down, now there should be 5 moves available and the
    #     # current tip for red should be at (1,0)
    #     next_state = grid.next_state(grid.spaces, (1,2))
    #     self.assertEqual(len(grid.possible_actions(next_state)), 5)
    #     self.assertEqual(Grid.get_flow_tip(next_state, grid.size, 1, grid.color_start_coords, grid.color_end_coords), (1,0))

    #     # Now try to move red back up, which it shouldn't be able to
    #     # do because that's where it just came from.
    #     with self.assertRaises(Exception) as context:
    #         grid.next_state(next_state, (1,0))
    #     self.assertTrue('Action is not viable' in str(context.exception))

# Program entry point.
if __name__ == "__main__":
	unittest.main()

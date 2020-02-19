import numpy as np
from optparse import OptionParser
from models.grid import Grid
import unittest

class TestGridFunctions(unittest.TestCase):

    # Test to make sure that the static Grid function |is_valid_state| works
    # as intended.
    def test_valid_states(self):
        return
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


    # Test to make sure that we can transition states correctly.
    # This also tests breaking flows, when a flow for a particular
    # color overlaps another flow.
    def test_next_state_function(self):
        return
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

        state = grid.start_state
        state = state.next_state((1,2))

        next_spaces = np.array([[1,2,3,0],
                                [1,0,0,0],
                                [0,0,0,0],
                                [1,2,3,0]])
        self.assertTrue(np.array_equal(state.spaces, next_spaces))
        self.assertEqual(state.tips[1], (1,0))


        state = state.next_state((1,1))
        next_spaces = np.array([[1,2,3,0],
                                [1,1,0,0],
                                [0,0,0,0],
                                [1,2,3,0]])
        self.assertTrue(np.array_equal(state.spaces, next_spaces))
        self.assertEqual(state.tips[1], (1,1))

        # Have 2 move down, obliterating 1s flow.
        state = state.next_state((2,2))
        next_spaces = np.array([[1,2,3,0],
                                [0,2,0,0],
                                [0,0,0,0],
                                [1,2,3,0]])
        self.assertTrue(np.array_equal(state.spaces, next_spaces))
        self.assertEqual(state.tips[1], (0,0)) 
        self.assertEqual(state.tips[2], (1,1)) 


    def test_multiple_transitions(self):
        return
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


        state = grid.start_state
        state = state.next_state((3,2))
        self.assertEqual(state.tips[3], (1,2))
        self.assertTrue(state.is_valid())

        state = state.next_state((3,3))
        self.assertEqual(state.tips[3], (1,1))
        self.assertTrue(state.is_valid())

        state = state.next_state((3,3))
        self.assertEqual(state.tips[3], (1,0))
        self.assertTrue(state.is_valid())

        state = state.next_state((3,2))
        print("STATE:")
        print(state.spaces)
        self.assertEqual(state.tips[3], (2,0))
        self.assertTrue(state.is_valid())
 

    def test_about_to_win_state(self):
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


        spaces = np.array([[ 1,  2,  3,  3],
                           [ 1,  2,  3,  3],
                           [ 1,  2,  3,  3],
                           [ 1,  2,  3,  3]])

        tips = {
            1:(2,0),
            2:(2,1),
            3:(3,3),
        }

        state = Grid.State(grid, spaces,tips)
        self.assertFalse(state.is_winning())
        possible_actions = state.possible_actions()
        self.assertEqual(len(possible_actions), 6)  

        # Get ourselves to the winning state. There shouldn't
        # be any more possible actions left.
        state = state.next_state((1,2))
        state = state.next_state((2,2))
        state = state.next_state((3,3))
        possible_actions = state.possible_actions()
        self.assertEqual(len(possible_actions), 0)
        self.assertTrue(state.is_winning())


    def test_basic_grid(self):
        # Create grid.
        grid = Grid(filename="levels/test_level.txt")

        all_states = grid.generate_all_states()
        winning = 0
        for state in all_states:
            winning = winning + state.is_winning()
            if state.is_winning():
                print(state.spaces)
        print("Num states: ", len(all_states))
        print("Num winning: ", winning)

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

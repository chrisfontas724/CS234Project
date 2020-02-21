import numpy as np
from optparse import OptionParser
from models.grid import Grid
import unittest
from models.vi_and_pi import value_iteration

class TestGridFunctions(unittest.TestCase):

    # Test to make sure that we can transition states correctly.
    # This also tests breaking flows, when a flow for a particular
    # color overlaps another flow.
    def test_next_state_function(self):
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
        #Create grid.
        grid = Grid(filename="levels/test_level.txt")

        all_states = grid.generate_all_states()
        print("Total states: ", len(all_states))
        winning = 0
        for state in all_states:
            winning = winning + state.is_winning()
            if state.is_winning():
                print(state.spaces)
        self.assertTrue(winning, 2)


        grid2 = Grid(filename="levels/test_level_2.txt")
        all_states = grid2.generate_all_states()
        print("Total states: ", len(all_states))
        winning = 0
        for state in all_states:
            winning = winning + state.is_winning()
            if state.is_winning():
                print(state.spaces)
        self.assertTrue(winning, 3)

        return

        # This takes a while to run, so remove the |return| up
        # above if you want to see it run too.
        grid3 = Grid(filename="levels/easy-1.txt")
        all_states = grid3.generate_all_states()
        print("Total states: ", len(all_states))
        winning = 0
        for state in all_states:
            winning = winning + state.is_winning()
            if state.is_winning():
                print(state.spaces)
        self.assertTrue(winning, 1)

    def test_all_4x4_grids(self):
        num_wins = 0
        for i in range(1, 16):
            grid = Grid(filename="levels/grid_" + str(i) + ".txt")
            vf, policy = value_iteration(grid)

            state = grid.start_state

            # We should expect each 4x4 grid to finish in less
            # than 100 turns
            turns = 0
            while turns < 100:
                state = state.next_state(policy[state])
    
                # Break if we're in the winning state.
                if state.is_winning():
                    print("Won " + "grid_" + str(i) + "!")
                    num_wins = num_wins + 1
                    break
                turns = turns + 1
        self.assertEqual(num_wins, 15)



# Program entry point.
if __name__ == "__main__":
	unittest.main()

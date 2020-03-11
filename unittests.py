import numpy as np
from optparse import OptionParser
from models.grid import Grid
import unittest
from models.vi_and_pi import value_iteration
from models.vi_and_pi import policy_iteration
from qlearning.qstate import QState

class TestGridFunctions(unittest.TestCase):


    def test_q_rewards(self):
        grid = Grid(filename="levels/4x4/grid_100.txt")

        state = QState(grid.start_state)


        print("Start state info: ")
        print("Spaces: ")
        print(state.state.spaces)
        print("Red start location: ", state.state.info.color_start_coords[1])
        print("Possible moves: ", state.state.possible_actions())
        print("Num Flows: ", state.completed_flow_count())
        print("Is (1,2) viable: ", state.is_viable_action((1,2)))

        result = state.step((1,2))
        print("Reward: ", result[1])
        print("")

        print("Next state: ")
        print("Num Flows: ", result[0].completed_flow_count())

        result = result[0].step((1,2))
        print("Reward: ", result[1])
        print("")

        result = result[0].step((1,2))
        print("TUPLE THIRD: ", result)
        print("Reward: ", result[1])
        print("")
        return


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


    def test_basic_grid_5x5(self):
        return
        #Create grid.
        grid = Grid(filename="levels/5x5/grid_10.txt")

        all_states = grid.generate_all_states()
        print("Total states for 5x5/grid_10: ", len(all_states))
        winning = 0
        for state in all_states:
            winning = winning + state.is_winning()
            if state.is_winning():
                print(state.spaces)
        self.assertTrue(winning >= 1)


        grid = Grid(filename="levels/5x5/grid_777.txt")

        all_states = grid.generate_all_states()
        print("Total states for 5x5/grid_777: ", len(all_states))
        winning = 0
        for state in all_states:
            winning = winning + state.is_winning()
            if state.is_winning():
                print(state.spaces)
        self.assertTrue(winning >= 1)


    def test_basic_grid(self):
        return
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


    def test_policy_iteration(self):
        return
        print("Policy iteration test...")
        num_wins = 0
        for i in range(1, 2):
            grid = Grid(filename="levels/grid_" + str(i) + ".txt")
            vf, policy = policy_iteration(grid)

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
        self.assertEqual(num_wins, 1)

    def test_all_4x4_grids(self):
        return

        num_wins = 0
        for i in range(1, 4):
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
        self.assertEqual(num_wins, 3)



# Program entry point.
if __name__ == "__main__":
	unittest.main()

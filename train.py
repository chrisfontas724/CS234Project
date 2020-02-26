import numpy as np
from optparse import OptionParser
from models.grid import Grid
from models.vi_and_pi import value_iteration
from renderer.renderer import GridRenderer
from models.vi_and_pi import policy_iteration
import pickle
import random

# Use the OptionParser library to get command line arguments
# for us, such as the file we want to load in.
def get_options():
    parser = OptionParser(usage="usage: %prog -l filename",
                          version="%prog 1.0")

    # Determines if we should use policy iteration or value iteration for training.
    parser.add_option("-a", "--algorithm",
                      action="store", # optional because action defaults to "store"
                      dest="algorithm",
                      default="value_iteration",
                      help="Pick an algorithm to use to train the policy",)

    # Deterimines if we should train, test, or one shot.
    parser.add_option("-m", "--mode",
                      action="store", # optional because action defaults to "store"
                      dest="mode",
                      default="train",
                      help="Pick a mode, train or test",)

    # File to load/save a run to.
    parser.add_option("-f", "--file",
                      action="store", # optional because action defaults to "store"
                      dest="file",
                      help="Load a policy from disk if it exists and save to this file.",)

    return parser.parse_args()


def play_game(grid, policy, max_turns=100):
  counter = 0
  state = grid.start_state
  while counter < max_turns:
    print("Tick!")
    action = None
    if not state in policy:
      all_actions = state.possible_actions()
      if len(all_actions) <= 0:
        print("No possible actions")
        return False
      print("Random action")
      action = random.choice(all_actions)
    # Get the action directly from the policy
    else:
      print("Action from policy")
      action = policy[state]

    if action is None:
      raise Exception("Action is none!")

    print("ACTION: " + str(action))
    state = state.next_state(action)
    if state.is_winning():
      return True
  return False

# Training script.
def main():
    # Grab the command line options.
    options, args = get_options()

    # Train by iterating over all grids in the training set, while using the
    # previous iteration's value function as the starting point for the next
    # iteration's training. The idea is that different boards of the same size
    # would be able to share states and ths would thus allow us to generalize
    # better to unseen boards.
    if options.mode == "train":
      algorithm = policy_iteration if options.algorithm == "policy_iteration" else value_iteration
      value = dict()
      policy = dict()
      for i in range(1,16): 
        print("Training iteration " + str(i))
        grid = Grid(filename="levels/grid_" + str(i) + ".txt")
        new_value, policy = algorithm(grid, policy, value)
        value = new_value.copy()

      # Save the result of training to a pickle file.
      with open("policies/" + options.file + ".pickle", 'wb') as handle:
        pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # If we're in test mode, then we load up an existing policy, and have it play
    # boards numbered 16-25. We count up how many of those 10 boards it wins.
    elif options.mode == "test":
      # Load up the pickle file we saved to during training.
      with open("policies/" + options.file + ".pickle", 'rb') as handle:
        policy = pickle.load(handle)

      print("Num keys: ", len(policy.keys()))

      num_wins = 0
      for i in range(16, 25):
        print("Playing game " + str(i))
        grid = Grid(filename="levels/grid_" + str(i) + ".txt")
        num_wins += play_game(grid, policy)
      print("Testing won " + str(num_wins) + "out of 10 games!")

# Program entry point.
if __name__ == "__main__":
	main()

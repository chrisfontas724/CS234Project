import numpy as np
from optparse import OptionParser
from models.grid import Grid
from models.vi_and_pi import value_iteration
from renderer.renderer import GridRenderer
from models.vi_and_pi import policy_iteration
import pickle

# Use the OptionParser library to get command line arguments
# for us, such as the file we want to load in.
def get_options():
    parser = OptionParser(usage="usage: %prog -l filename",
                          version="%prog 1.0")

    # Determines if we should use policy iteration or value iteration for training.
    parser.add_option("-a", "--algorithm",
                      action="store", # optional because action defaults to "store"
                      dest="algorithm",
                      default="policy_iteration",
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
        grid = Grid(filename="levels/grid_" + str(i) + ".txt")
        new_value, policy = algorithm(grid, value)
        value = new_value.copy()

      # Save the result of training to a pickle file.
      with open("policies/" + options.file + ".pickle", 'wb') as handle:
        pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif options.mode == "test":
      # Load up the pickle file we saved to during training.
      with open("policies/" + options.file + ".pickle", 'rb') as handle:
        b = pickle.load(handle)

# Program entry point.
if __name__ == "__main__":
	main()

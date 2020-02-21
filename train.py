import numpy as np
from optparse import OptionParser
from models.grid import Grid
from models.vi_and_pi import value_iteration
from renderer.renderer import GridRenderer
from models.vi_and_pi import policy_iteration

# Use the OptionParser library to get command line arguments
# for us, such as the file we want to load in.
def get_options():
    parser = OptionParser(usage="usage: %prog -l filename",
                          version="%prog 1.0")

    parser.add_option("-l", "--level",
                      action="store", # optional because action defaults to "store"
                      dest="level",
                      default="test_level.txt",
                      help="FlowFree level to load",)

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
                      default="one_shot",
                      help="Pick a mode, train or test",)

    # File to load/save a run to.
    parser.add_option("-f", "--file",
                      action="store", # optional because action defaults to "store"
                      dest="file",
                      help="Load a policy from disk if it exists and save to this file.",)

    return parser.parse_args()

def main():
    # Grab the command line options.
    options, args = get_options()

    # Instantiate the FLowFree grid.
    grid = Grid(filename="levels/" + options.level)

    # Initialize the renderer.
    renderer = GridRenderer(options.level)

    if options.mode == "train":
      pass
    elif options.mode == "test":
      pass

# Program entry point.
if __name__ == "__main__":
	main()

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


    return parser.parse_args()

# Use to test a single level using either policy iteration or value iteration.
def main():
    # Grab the command line options.
    options, args = get_options()

    # Instantiate the FLowFree grid.
    grid = Grid(filename="levels/" + options.level)

    # Initialize the renderer.
    renderer = GridRenderer(options.level)

    print("Algorithm: ", options.algorithm)

    value_function, policy = policy_iteration(grid) if options.algorithm == "policy_iteration" else value_iteration(grid)
    print("Completed iteration!")

    # Now let's try out our policy!
    state = grid.start_state
    while True:
      state = state.next_state(policy[state])
      
      # Draw the grid to the screen.
      renderer.render(state)

      # Break if we're in the winning state.
      if state.is_winning():
        break

    # Close the window.
    renderer.tear_down()

# Program entry point.
if __name__ == "__main__":
	main()

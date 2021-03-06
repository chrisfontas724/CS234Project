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

    return parser.parse_args()

# Just a very simple script to render a single grid, for sanity checking visually.
def main():
    # Grab the command line options.
    options, args = get_options()

    # Instantiate the FLowFree grid.
    grid = Grid(filename="levels/" + options.level)

    # Initialize the renderer.
    renderer = GridRenderer(options.level)
      
    # Draw the grid to the screen.
    renderer.render(grid.start_state)

    # Close the window.
    renderer.tear_down()

# Program entry point.
if __name__ == "__main__":
	main()

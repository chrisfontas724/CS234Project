import numpy as np
import cv2
from optparse import OptionParser
from models.grid import Grid
from renderer.renderer import Renderer

# Use the OptionParser library to get command line arguments
# for us, such as the file we want to load in.
def get_options():
    parser = OptionParser(usage="usage: %prog -l filename",
                          version="%prog 1.0")

    parser.add_option("-l", "--level",
                      action="store", # optional because action defaults to "store"
                      dest="level",
                      default="easy-1.txt",
                      help="FlowFree level to load",)
    return parser.parse_args()

def main():

    # Grab the options.
    options, args = get_options()

    # Instantiate the FLowFree grid.
    grid = Grid(filename="levels/" + options.level)

    # Initialize the renderer
    renderer = Renderer()

if __name__ == "__main__":
	main()

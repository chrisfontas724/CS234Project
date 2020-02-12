import numpy as np
import cv2
from optparse import OptionParser
from models.grid import Grid

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

    # Test rendering a dummy image of a circle.
    img = np.zeros((512,512,3), np.uint8)
    img = cv2.circle(img,(447,63), 63, (0,0,255), -1)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyWindow('image')

if __name__ == "__main__":
	main()

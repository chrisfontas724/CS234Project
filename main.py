import numpy as np
import cv2
from optparse import OptionParser
from models.grid import Grid


color_dictionary = {
  # TODO - Fill this out
}


def loadfile(filename):
	file = open(filename, "r") 
	print(file.read())


def main():
    print("Hello, world!")

    parser = OptionParser(usage="usage: %prog -l filename",
                          version="%prog 1.0")

    parser.add_option("-l", "--level",
                      action="store", # optional because action defaults to "store"
                      dest="level",
                      default="easy-1.txt",
                      help="FlowFree level to load",)
    (options, args) = parser.parse_args()
    print("Loading level ", options.level)


    loadfile("levels/" + options.level)


    # Test rendering a dummy image of a circle.
    img = np.zeros((512,512,3), np.uint8)
    img = cv2.circle(img,(447,63), 63, (0,0,255), -1)
    cv2.imshow('image',img)
    cv2.waitKey(0)

if __name__ == "__main__":
	main()

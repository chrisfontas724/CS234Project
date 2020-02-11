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


if __name__ == "__main__":
	main()

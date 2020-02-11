import numpy as np
from optparse import OptionParser


def main():
    print("Hello, world!")

    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")

    parser.add_option("-l", "--level",
                      action="store", # optional because action defaults to "store"
                      dest="level",
                      default="easy-1.txt",
                      help="FlowFree level to load",)
    (options, args) = parser.parse_args()


    print("Loading level ", options.level)

if __name__ == "__main__":
	main()

import numpy as np
from optparse import OptionParser
from models.grid import Grid
from renderer.renderer import GridRenderer


def main():

    # Instantiate the FLowFree grid.
    grid = Grid(filename="levels/test_level.txt")

# Program entry point.
if __name__ == "__main__":
	main()

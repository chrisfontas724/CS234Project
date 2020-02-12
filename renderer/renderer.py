import numpy as np
from optparse import OptionParser
from models.grid import Grid, color_dict
import cv2

class GridRenderer:
    def __init__(self):
        print("Init renderer")
        # This is th size of the image to display in pixels.
        self.size= 1000
        self.img = np.zeros((self.size,self.size, 3), np.uint8)

    # This function takes in a grid and renders it to the screen
    # so that we can visually keep track of the progress being
    # made by our algorithm(s).
    def render(self, grid):

        pixels_per_tile = int(self.size / grid.size)

        for x in range(grid.size):
            for y in range(grid.size):
                item = int(grid.spaces[x][y])
                if item is not 0:
                    radius = int(pixels_per_tile/2)
                    x_offset = int(x * pixels_per_tile) + radius
                    y_offset = int(y * pixels_per_tile) + radius
                    print(color_dict[item])
                    self.img = cv2.circle(self.img,(x_offset, y_offset), int(pixels_per_tile/2), color_dict[item], -1)
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        cv2.destroyWindow('image')
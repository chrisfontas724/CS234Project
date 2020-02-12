import numpy as np
from optparse import OptionParser
from models.grid import Grid
import cv2

# This dictionary maps the numbers that represent
# the colored circles and their associated paths
# to actual RGB color values.
color_dict = {
  1:(255,0,0),     # Red
  2:(0,255,0),     # Green
  3:(0,0,255),     # Blue
  4:(255,255,0),   # Yellow
  5:(0,255,255),   # Turquoise
  6:(255,0,255),   # Purple
  7:(127,255,127),
  8:(255,127,127),
  9:(127,127,255),
  10:(0,127,127),
  11:(127,0,0),
  12:(127,0,127),
  13:(127, 127, 0),
  14:(50, 50, 127),

}

# This class visualizes a FlowFree grid. It is initialized with
# the name of the grid, and is currently hardcoded to have the
# dimensions (500,500) in pixels.
class GridRenderer:
    def __init__(self, name):
        print("Init renderer")
        # This is th size of the image to display in pixels.
        self.name = name
        self.size= 500
        self.img = np.zeros((self.size,self.size, 3), np.uint8)

    # This function takes in a grid and renders it to the screen
    # so that we can visually keep track of the progress being
    # made by our algorithm(s).
    def render(self, grid):

        pixels_per_tile = int(self.size / grid.size)

        # Draw the grid lines first.
        for x in range(grid.size):
            self.img = cv2.line(self.img, (x*pixels_per_tile,0), (x*pixels_per_tile,500), (255,255,0), 1)
            self.img = cv2.line(self.img, (0, x*pixels_per_tile), (500, x*pixels_per_tile), (255,255,0), 1)

        # Now draw the circles.
        for x in range(grid.size):
            for y in range(grid.size):
                item = int(grid.spaces[x][y])
                if item is not 0:
                    border = int(pixels_per_tile / 10)
                    radius = int((pixels_per_tile - border)/2)
                    x_offset = int((x + 0.5) * pixels_per_tile)
                    y_offset = int((y + 0.5) * pixels_per_tile) 
                    print(color_dict[item])
                    self.img = cv2.circle(self.img,(x_offset, y_offset), radius, color_dict[item], -1)
        cv2.imshow(self.name, self.img)
        cv2.waitKey(0)
        cv2.destroyWindow(self.name)


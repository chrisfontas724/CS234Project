import numpy as np
from optparse import OptionParser
from models.grid import Grid
import cv2

class GridRenderer:
    def __init__(self):
        print("Init renderer")
        self.img = np.zeros((512,512,3), np.uint8)


    # This function takes in a grid and renders it to the screen
    # so that we can visually keep track of the progress being
    # made by our algorithm(s).
    def render(self, grid):
         # Test rendering a dummy image of a circle.
        self.img = cv2.circle(self.img,(447,63), 63, (0,0,255), -1)
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        self.img = cv2.circle(self.img,(447,63), 63, (0,255,255), -1)
        cv2.imshow('image',self.img)
        cv2.waitKey(0)
        cv2.destroyWindow('image')
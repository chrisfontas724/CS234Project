import numpy as np
from optparse import OptionParser
from models.grid import Grid
import cv2

class Renderer:
    def __init__(self):
        print("Init renderer")
         # Test rendering a dummy image of a circle.
        img = np.zeros((512,512,3), np.uint8)
        img = cv2.circle(img,(447,63), 63, (0,0,255), -1)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        img = cv2.circle(img,(447,63), 63, (0,255,255), -1)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyWindow('image')
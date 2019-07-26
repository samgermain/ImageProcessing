import skimage.util as util
import skimage.io as io
import numpy as np
import os as os


# You can import other modules as needed.

def segleaf(I):
    '''
    Segment a leaf image.
    Use Thresholding. Use region processing after segmentation
    :param I: Color leaf image to segment.
    :return: Boolean image where True pixels represent foreground (i.e. leaf pixels).
    '''

    pass  # replace this with your return statement.  This is just a placeholder to prevent a syntax error.

myimage = io.imread('images/noisy-test/12003.png')
I = util.img_as_float(myimage)
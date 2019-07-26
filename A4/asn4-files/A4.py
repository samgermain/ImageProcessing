import skimage.util as util
import skimage.io as io
import numpy as np
import glob
import skimage.segmentation as seg
import skimage.filters as filt
import skimage.morphology as morph
import skimage.morphology.selem as selem
import os as os
import warnings
warnings.filterwarnings("ignore")

def segleaf(I):
    """
    Leaf segmentation using random walker
    :param I: Input image as a float

    :return: A binary image that shows the segmentation of the leaf from the background
    """
    #Separate the image into it's three colour components
    red = 255 * I[:, :, 0]
    green = 255 * I[:, :, 1]
    blue = 255 * I[:,:,2]

    #Create a new image with the amount of greeness exaggerrated
    greenness = 1.97 * green - red - blue
    #Find a threshold value
    T = filt.threshold_otsu(greenness)
    #Create an array of identical size to the image, with seed pixels labelled as fg and bg
    seeds = np.zeros_like(I, dtype=np.uint8)
    seeds[greenness >= T] = 1  # fg label
    try:
        seeds[greenness < 5] = 2  # bg label
        labels = seg.random_walker(I, seeds, beta=10, multichannel=True)
    except:
        #For when a IndexError occurs
        try:
            seeds[greenness < 10] = 2  # bg label
            labels = seg.random_walker(I, seeds, beta=10, multichannel=True)
        except:
            seeds[greenness < 22] = 2  # bg label
            labels = seg.random_walker(I, seeds, beta=10, multichannel=True)

    #Label the bg labelled pixels as False
    labels[labels == 2] = 0
    labels = labels[:,:,0]  #All three colour components share the same value now, so only one is needed.

    #Perform some region processing to improve the result
    labels = morph.opening(labels, selem.diamond(1))
    labels = morph.binary_closing(labels, selem.diamond(2.5))
    labels = morph.remove_small_objects(labels, 30)
    labels = morph.binary_closing(labels, selem.diamond(3))
    labels = morph.remove_small_objects(labels, 340)
    labels = morph.binary_closing(labels, selem.diamond(6))
    return labels

def dice_coefficient(bwA, bwG):
    '''
    Dice coefficient between two binary images
    :param bwA: a binary (dtype='bool') image
    :param bwG: a binary (dtype='bool') image
    :return: the Dice coefficient between them
    '''
    intersection = np.logical_and(bwA, bwG)

    return 2.0*np.sum(intersection) / (np.sum(bwA) + np.sum(bwG))


def Driver():
    '''
    Performs the Random Walker Segmentation on each of the noisy image files and does a DSC comparison with each equivelent ground truth image
    :return: Nothing
    '''
    images = [os.path.basename(x) for x in glob.glob('noisyimages/*.png')]
    avD = 0
    length = len(images)
    for i in range(0,length):
        I = util.img_as_float(io.imread('noisyimages/' + images[i]))
        J = util.img_as_bool(io.imread('groundtruth/thresh' + images[i]))
        A = segleaf(I)
        dsc = dice_coefficient(A, J)
        print(images[i])
        print("DSC: ", dsc)
        if (dsc > 0.6):
            print("Recognized as a leaf")
        avD += dsc
    print("Average DSC: ", avD/length)

Driver()
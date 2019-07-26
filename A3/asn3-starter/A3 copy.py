import skimage.util as util
import skimage.io as io
import numpy as np
import glob
import skimage.segmentation as seg
import scipy.spatial.distance as distance
import skimage.filters as filt
import skimage.morphology as morph
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import mark_boundaries
import os as os
import matplotlib.pyplot as plt
#import cv2
import skimage.color as color

# You can import other modules as needed.

def otsu(I):
    '''
    Finds the optimal threshold of an image
    :param I: Input image to find the threshold
    :return:A tuple of values that minimizes weighted sum of group variances σ^(2)w(t) for the red and green channel
    '''

    #Isolating the colour channels and placing them in bins
    red = 256 * I[:, :, 0]  # Zero out contribution from green
    green = 256 * I[:, :, 1]
    gt = filt.threshold_otsu(green)
    rt = filt.threshold_otsu(red)
    return (gt,rt)

def segleaf(I):
    '''
    Segment a leaf image.
    Use Thresholding. Use region processing after segmentation
    :param I: Color leaf image to segment.
    :return: Boolean image where True pixels represent foreground (i.e. leaf pixels).
    '''

    red = 255 * I[:, :, 0]
    green = 255 * I[:, :, 1]
    blue = 255 * I[:,:,2]
    greenness = 2 * green - red - blue
    T = filt.threshold_otsu(greenness)
    return greenness > T

def DSC(A,G):
    '''
    Takes a thresholded binary image, and a ground truth img, and computes the Dice Similarity Coeffiencient
    :param A: The thresholded binary image
    :param G: The ground truth img
    :return: The Dice Similarity Coefficient, with numbers close to 1 meaning the two images are nearly identical and numbers close to 0 meaning the two images have little in common
    '''
    #reshape A and G to 1D arrays
    A2 = np.reshape(A, [A.shape[0]*A.shape[1]])
    G2 = np.reshape(G, [G.shape[0]*G.shape[1]])
    sim = aTot = gTot = 0   #sim: the pixels shared by both A and G, aTot: The total true A pix, gTot: the total true G pix
    for reg in range(0,len(A2)):
        #If the two images have this pixel as true
        if A2[reg] == True and G2[reg] == True:
            sim += 1
            aTot += 1
            gTot += 1
        elif A2[reg] == True:
            aTot += 1
        elif G2[reg] == True:
            gTot += 1
    return 2*sim/(aTot + gTot) #Dice similarity coefficient formula

def MSD(A,G):
    '''
    Takes a thresholded binary image, and a ground truth img(binary), and computes the mean squared absolute difference
    :param A: The thresholded binary image
    :param G: The ground truth img
    :return:
    '''
    A_con = np.transpose(np.vstack(np.where(seg.find_boundaries(A==True))))
    G_con = np.transpose(np.vstack(np.where(seg.find_boundaries(G==True))))
    sum = 0
    for aPoint in A_con:
        min = 9999999
        for gPoint in G_con:
            e = ((aPoint[0] - gPoint[0]) + (aPoint[1] - gPoint[1]))**2
            if e < min:
                min = e
        sum += min
    return sum/(A_con.shape[1])



def HS(A,G):
    '''
    :param A:
    :param G:
    :return:
    '''

    A_con = np.transpose(np.vstack(np.where(seg.find_boundaries(A == True))))
    G_con = np.transpose(np.vstack(np.where(seg.find_boundaries(G == True))))
    max1 = 0
    for aPoint in A_con:
        min = 999999
        for gPoint in G_con:
            e = ((aPoint[0] - gPoint[0]) + (aPoint[1] - gPoint[1])) ** 2
            if e < min:
                min = e
        if (min > max1):
            max1 = e
    max2 = 0
    for gPoint in G_con:
        min = 9999999
        for aPoint in A_con:
            e = ((gPoint[0] - aPoint[0]) + (gPoint[1] - aPoint[1])) ** 2
            if e < min:
                min = e
        if (min > max2):
            max2 = e

    return max(max1,max2)

def Driver():
    images = [os.path.basename(x) for x in glob.glob('images/*.png')]
    avD = 0
    avS = 0
    avH = 0
    length = len(images)
    for i in range(0,length):
        I = util.img_as_float(io.imread('images/' + images[i]))
        J = util.img_as_bool(io.imread('groundtruth/thresh' + images[i]))
        A = segleaf(I)
        dsc = DSC(A, J)
        msd = MSD(A, J)
        hs = HS(A, J)
        print(images[i])
        print("DSC: ", dsc)
        print("MSD: ", msd)
        print("HS: ", hs)
        if (dsc > 0.6):
            print("Recognized as a leaf")
        avD += dsc
        avS += msd
        avH += hs
    print("Average DSC: ", avD/length)
    print("Average MSD: ", avS/length)
    print("Average HS: ", avH/length)

#Driver()
image = io.imread('images/image_0090.png')
I = util.img_as_float(image)
J = util.img_as_bool(io.imread('groundtruth/threshimage_0090.png'))
#print(otsu(I))
A = segleaf(I)
#plt.imshow(A)
#plt.show()
#boundaries = seg.find_boundaries(A, connectivity=2, mode='inner ')
L = morph.label(A, connectivity=2)
boundaries = seg. find_boundaries (L, connectivity =2,mode ='inner ')
#L = np.transpose(np.vstack(np.where(seg.find_boundaries(A == True))))
img_w_b = seg.mark_boundaries(image, L, color=(1 ,1 ,0))
plt.imshow(img_w_b)
plt.show(img_w_b)
#Driver()
#print(DSC(A,J))
#print(MSD(A,J))
#Driver()


#print greyscale histogram
#g_img = color.rgb2gray(I)
#hist = plt.hist(g_img, bins='auto')
#plt.plot(hist)
#plt.show()
#img = cv2.imread('images/image_0011.png',0)
#plt.hist(img.ravel(),256,[0,256]); plt.show()

#print the rgb histogram
#color = ('b','g','r')
#for i,col in enumerate(color):
#    histr = cv2.calcHist([img],[i],None,[256],[0,256])
#    plt.plot(histr,color = col)
#    plt.xlim([0,256])
#plt.show()
#I = util.img_as_float(myimage)
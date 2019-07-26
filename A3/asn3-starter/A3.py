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

def otsu_helper(counts):
    '''
    Does the arithmetic of the otsu method
    :param counts: An array recording how many pixels are in each bin
    :return: A threshold found by minimizing the weighted sum of group variances σ^(2)w(t)
    '''

    # variables for otsu's method
    m1t = 0  # mean prob for first group
    q1t = 0  # prob that a given greylevel < t
    s21t = 0  # variace for the group of values less that t
    tot = 0 #keep track of the total amount of pixels captured to bins so far
    pix = sum(counts)   #The total number of pixels
#    print(counts)
    group1 = np.zeros(shape=(256,2))
    #Calculate the mean probability and variances for group1 for all values of t
    for i in range(0,256):
        j = i+1
        bin = counts[i]
        #print(i,":",bin)
        tot += bin
        pi = bin/pix
        q1t += tot/pix
        if (q1t != 0):
            m1t += (j * pi )/q1t
            s21t += ( (j - m1t)**2 * pi )/q1t
            group1[i] = [q1t,s21t]  # Store the probability and variance for later use in finding the sum of group variances
        else:
            group1[i] = [None,None]
    tot = 0
    q2t = 0  # prob that a given greylevel > t
    m2t = 0  # mean prob for second group
    s22t = 0  # variace for the group of values greater than t
    group2 = np.zeros(shape=(256,2))
    group2[255] = [None,None] #At threshold 256, there are no values greater than t
    # Calculate the mean probability and variances for group2 for all values of t
    for i in range(254, -1, -1):
        j = i+1
        bin = counts[j]   #Because all the values from t+1 to G are summed
        tot += bin
        pi = (bin / pix)
        q2t += tot / pix
        if (q2t != 0):
            m2t += (j*pi)/q2t
            s22t += ((j - m2t)**2 * pi)/q2t
            group2[i] = [q2t, s22t]
        else:
            group2[i] = [None, None]

    s2w = 999999999999  # Weight sum of group variances
    t = 0   #The optimal threshold value
    #Calculate the weighted sum of group variances for all values of t and save the minimum
    for i in range(0,256):
        if (not np.isnan(group1[i,1]) and not np.isnan(group2[i,1])):
            s2wt = group1[i,0]*group1[i,1] + group2[i,0]*group2[i,1]
 #           print(i,":","{0:.{1}e}".format(s2wt, 1))
            if s2w > s2wt:
                s2w = s2wt
                t = i

    return t


def otsu(I):
    '''
    Finds the optimal threshold of an image
    :param I: Input image to find the threshold
    :return:A tuple of values that minimizes weighted sum of group variances σ^(2)w(t) for the red and green channel
    '''

    #Isolating the colour channels and placing them in bins
    red = 256 * I[:, :, 0]  # Zero out contribution from green
    green = 256 * I[:, :, 1]
    blue = 256 * I[:, :, 2]
    #red, green, blue = np.moveaxis(I, -1, 0)
    bins = np.array(range(0,257))
    g_counts,pixels = np.histogram(green,bins)
    r_counts,pixels = np.histogram(red, bins)
    b_counts,pixels = np.histogram(blue,bins)
#    print('gt = otsu_helper(g_counts)')
    #gt = otsu_helper(g_counts)
    gt = filt.threshold_otsu(green)

#    print('rt = otsu_helper(r_counts)')
    #rt = otsu_helper(r_counts)
    rt = filt.threshold_otsu(red)
#    return (gt,rt)
#   Creating a histogram of the green colour channel
#    pixels = pixels[:-1]
#    plt.bar(pixels, g_counts, align='center')
#    plt.xlim(-1, 256)
#    plt.show()
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

    #gt, rt = otsu(I)
    #print(gt, rt)

    #bImage = np.zeros(shape=[I.shape[0],I.shape[1]], dtype=bool)

    greenness = 2 * green - red - blue
    T = filt.threshold_otsu(greenness)
    return greenness > T


    # plt.figure()
    # plt.imshow(greenness, cmap='gray')
    # plt.show()


    # for i in range(0,I.shape[0]-1):
    #     for j in (0,I.shape[1]-1):
    #         if (green[i,j] > gt and red[i,j] < rt):
    #             bImage[i,j] = True

    return bImage  # replace this with your return statement.  This is just a placeholder to prevent a syntax error.

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
  #      print(A2[reg],G2[reg])
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
#    for i in range(0,sim2.shape[0]):
 #       for j in range(0,sim2.shape[1]):
  #          min = 9999999
   #         for k in range(0,sim2.shape[0]):
    #            for l in range(0,sim2.shape[1]):
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

    #for i in range(0, A.shape[0]):
    #    for j in range(0, A.shape[1]):
    #        if (A[i, j] == True):
    #            min = 9999999
    #            for k in range(0, G.shape[0]):
    #                for l in range(0, G.shape[1]):
    #                    if (G[k, l] == True):
    #                        e = abs(i - k) + abs(j - l)
    #                        if e < min:
    #                            min = e
    #            if (min>max1):
    #                max1 = e
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
        #msd = MSD(A, J)
        #hs = HS(A, J)
        print(images[i])
        print("DSC: ", dsc)
        #print("MSD: ", msd)
        #print("HS: ", hs)
        if (dsc > 0.6):
            print("Recognized as a leaf")
        avD += dsc
        #avS += msd
        #avH += hs
    print("Average DSC: ", avD/length)
    #print("Average MSD: ", avS/length)
    #print("Average HS: ", avH/length)

#Driver()
I = util.img_as_float(io.imread('images/image_0090.png'))
J = util.img_as_bool(io.imread('groundtruth/threshimage_0090.png'))
#print(otsu(I))
A = segleaf(I)
plt.imshow(A)
plt.show()
#print(DSC(A,J))
#print(MSD(A,J))
Driver()


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
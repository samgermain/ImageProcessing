import numpy as np
# np.set_printoptions(threshold=np.nan)
import skimage.util as util
import skimage.io as io
import skimage.morphology as morph
import skimage.segmentation as seg
import cv2


# Code your HoCS function here
#def outline(b):
#    im2, contours, hierarchy = cv2.findContours(b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    cim = np.zeros_like(b)
#    cv2.drawContours(cim, contours, -1, 255, 1)
#def normalize(a):
#    if (a>0.5):
#        return a-0.1
#    elif (a<0.5):
#        return a+0.1
from sklearn.preprocessing import normalize

def normalize(v):
#    norm1 = x / np.linalg.norm(x)
#    norm2 = normalize(x[:, np.newaxis], axis=0).ravel()
#    return np.all(norm1 == norm2)
    return (v - min(v))/(max(v)-min(v))

def HoCS(B, min_scale, max_scale, increment, num_bins):
    '''
     Computes a histogram of curvature scale for the shape in the binary image B.
    Boundary fragments due to holes are ignored.
    :param B: A binary image consisting of a single foreground connected component.
    :param min_scale: smallest scale to consider (minimum 1)
    :param max_scale: largest scale to consider (max_scale > min_scale)
    :param increment:  increment on which to compute scales between min_scale and max_scale
    :param num_bins: number of bins for the histogram at each scale
    :return: 1D array of histograms concatenated together in order of increasing scale.
    '''

    #print(B)
    #outline = np.logical_and(B,morph.binary_erosion(B))  # Get the border pixels of the binary image
    outline = seg.find_boundaries(B, connectivity=2, mode='inner')
    #outlin = outline(B)
    #print("outline:", outline)
    # Save all the outline coordinates
    o_index = np.argwhere(outline == True)  # 2d array of two columns, row# and col# in each column
    #print(B[o_index[0][0]][o_index[0][1]])
    #print(o_index)
    # Create a 2d histogram to save of the histogram of curvature scales
    scales = int(((max_scale - min_scale) / increment) + 1)
    final_array = np.array([])
    #i = 0
    pd = 500
    B2 = np.pad(B,((pd,pd),(pd,pd)),'edge')

    # Loop through each scale
    for scale in range(min_scale, max_scale+1, increment):
        # print(o_index[:][0])
        hist_cur = np.zeros(len(o_index))
        disk = morph.disk(scale, dtype=bool)
        num_pix = np.count_nonzero(disk == 1)
        for pix in range(0, len(o_index)):
            # calculate how many 1 pixels in radius at o_index[pix][0],o_index[pix][1]
            # Calculate the normalized area integral invariant
            l = o_index[pix][0] - scale  # left
            r = o_index[pix][0] + scale  + 1# right
            u = o_index[pix][1] - scale  # above
            d = o_index[pix][1] + scale  + 1# below
            #print("l:", l, "r:", r, "u:", u, "d:", d)
            #print(disk)
            nbrhd = B2[l+pd:r+pd,u+pd:d+pd]  # neighboard about the outline pixel
            #print(nbrhd)
            overlap = np.logical_and(nbrhd, disk)  # fg pixels in the circle
            fg = np.count_nonzero(overlap == 1)
            kp = fg / (num_pix)  # normalized area integral invarient
            hist_cur[pix] = kp
            #print("fg:", fg, "kp:", kp)
            #if (pix == len(o_index)-1):
            #    hist_cur = normalize(hist_cur)

        # Save the histogram of curvature for this scale
    #    print(hist_cur)
        hist, other_stuff = np.histogram(hist_cur, bins=num_bins, range=(0.0,1.0))
    #    print(hist)
        #divide the histogram by the sum of all the bins
        final_array = np.concatenate([final_array,hist])
        #np.delete(final_array,0)
        #i += 1
    # reshape to a 1d array, this will be the concatted histograms
    return final_array / int(o_index.shape[0])

import skimage.io as io
import skimage.util as util
import matplotlib.pyplot as plt
#% matplotlib inline

B = util.img_as_bool(io.imread('leaftraining/threshimage_0001.png'))
y = HoCS(B, 5, 25, 10, 10)
x = range(0,len(y))
plt.bar(x,height=y)
plt.ylim(0,1.0)
#plt.show()

#A1S3
import glob
import os
import random

images = [os.path.basename(x) for x in glob.glob('leaftraining/*.png')]
images.sort()
length = len(images)

min = 5
max = 25
inc = 8
bins = 5

hocs_feat = []
for i in range(0, length):
    B = util.img_as_bool(io.imread('leaftraining/' + images[i]))
    hocs = HoCS(B, min, max, inc, bins)
    hocs_feat.append(hocs)
hocs_feat = np.asarray(hocs_feat)

labels = np.zeros(length)
clazz = 1
for i in range(0,length):
    if (i==10):
        clazz = 2
    elif (i==20):
        clazz = 3
    labels[i] = clazz

#A1S4
import sklearn.neighbors as neighbors
neighbs = 3
knn = neighbors.KNeighborsClassifier(n_neighbors=neighbs)
knn.fit(hocs_feat, labels)

#A1S5
images = [os.path.basename(x) for x in glob.glob('leaftesting/*.png')]
length = len(images)

hocs_feat = []
for i in range(0, length):
    B = util.img_as_bool(io.imread('leaftesting/' + images[i]))
    hocs = HoCS(B, min, max, inc, bins)
    hocs_feat.append(hocs)
hocs_feat = np.asarray(hocs_feat)

labels = knn.predict(hocs_feat)

#A1S6
true_labels = np.zeros(length)
clazz = 1
for i in range(0,length):
    if (i==50):
        clazz = 2
    elif (i==77):
        clazz = 3
    true_labels[i] = clazz

confusion = np.zeros((clazz,clazz))
for i in range(0, len(true_labels)):
    print("lt", true_labels[i], "l", labels[i])
    lt = int(true_labels[i])
    l = int(labels[i])
    confusion[lt-1, l-1] += 1

print("Confusion Matrix\n", confusion)

print("Classification rate: ", np.trace(confusion)/np.sum(confusion) * 100)


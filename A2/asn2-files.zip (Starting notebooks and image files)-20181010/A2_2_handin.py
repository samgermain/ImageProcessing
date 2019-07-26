import skimage.io as io
import skimage.util as util
import skimage.feature as feat
import matplotlib.pyplot as plt

# Write your code here.  It shouldn't take much.  Adapt the example on the last slide of the Topic 5 lecture notes.

import skimage.data as data
import skimage.color as color
import math as m
import skimage.feature as feat

image = io.imread('lentil.png')
I = util.img_as_float(image)
image_gray = color.rgb2gray(I)
blobs_log = feat.blob_log(image_gray, max_sigma =30, num_sigma =10, threshold =.1)
blobs_log[:, 2] = blobs_log[:, 2] * m.sqrt(2)
plt.figure ()
plt.imshow(I)
for row in blobs_log:
    c = plt.Circle((row [1], row[0]), row[2], color='lime', linewidth=1, fill=False)
    plt.gca().add_patch(c)
    plt.tight_layout
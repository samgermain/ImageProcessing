import skimage.io as io
import matplotlib.pyplot as plt
import skimage.color as color
import skimage.util as util
import skimage.exposure as exp
import numpy as np

plt.rcParams['image.cmap'] = 'gray'
myimage = io.imread('asn0-files/parrot.png')
greyscale_img = color.rgb2gray(myimage)
float_img = color.rgb2gray(greyscale_img)
width, height = float_img.shape
plt.title('Parrot')
plt.imshow(float_img, vmin=0, vmax=1)
plt.show()
plt.subplots(2,2)
p1 = float_img[:width//2, :height//2]
plt.subplot(2,2,1)
plt.imshow(p1)
p2 = float_img[:width//2, height//2:]
plt.subplot(2,2,2)
plt.imshow(p2)
p3 = float_img[width//2:, :height//2]
plt.subplot(2,2,3)
plt.imshow(p3)
p4 = float_img[width//2:, height//2:]
plt.subplot(2,2,4)
plt.imshow(p4)
plt.show()

#thresholding
thresh = 0.5
binary = float_img > thresh
plt.imshow(binary, cmap=plt.cm.gray)
plt.title('Thresholded Parrot')
plt.show()
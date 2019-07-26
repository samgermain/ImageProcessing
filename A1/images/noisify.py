import os as os
import skimage.io as io
import skimage.util as util
import skimage.filters as filt
import skimage.color as color
import skimage.exposure as exposure
import scipy.ndimage as nd
import numpy as np
import random as rand

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

import matplotlib.pyplot as plt
import skimage.morphology as morph

for root, dirs, files in os.walk('noiseless'):
    for filename in files:
        print(filename)
        if filename == '.DS_Store':
            continue

        I = util.img_as_float(io.imread(os.path.join(root, filename)))

        #io.imsave(os.path.join(root, filename[:-4]+'.png'), I)


        spikenoise = util.random_noise(np.zeros(I.shape[0:2]), mode="s&p", amount=.02, salt_vs_pepper=1.0)
        # sigma = rand.uniform(0, 1)
        # G = matlab_style_gauss2D(sigma=sigma,shape=(9,9))
        # G = G / np.max(G)
        #spikenoise = nd.filters.correlate(spikenoise, G)


        I_gray = color.rgb2gray(I)
        spikenoise = spikenoise * (I_gray < .5)
        spikenoise = color.gray2rgb(spikenoise)

        noisy = spikenoise+I
        noisy = util.random_noise(noisy, 'gaussian', var=rand.uniform(0.001, 0.005))

        io.imsave(os.path.join('noisy', filename[:-4]+'.png'), np.clip(noisy, 0, 1))




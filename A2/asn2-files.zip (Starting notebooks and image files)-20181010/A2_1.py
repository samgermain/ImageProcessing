import skimage.util as util
import numpy as np
import scipy.stats
#import matplotlib as plt
import skimage.io as io
import skimage.filters as filt
from numpy import array
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

from PIL import Image

#Q1
def color_dot_product(A, B):
    '''
    Element-by-element dot product in a 2D array of vectors.

    :return: An array in which index [i,j,:] is the dot product of A[i,j,:] and B[i,j,:].
    '''
    return np.sum(A.conj() * B, axis=2)

    #:param I: An image as a float
    #:return: A tuple containing theta(the direction of the largest colour change, followed by FTheta(the magnitude of the rate of change))


def color_sobel_edges(I):
    # Separate red, blue and green channels

    #red = I[:, :, 0]  # Zero out contribution from green
    #I_red[:, :, 2] = 0  # Zero out contribution from blue
    #h = filt.sobel_h(I_red)
    #v = filt.sobel_v(I_red)
    #red_theta = np.arctan2(v, h)
    #green = I[:,:,1]
    #I_blue[:, :, 1] = 0  # Zero out contribution from green
    #I_blue[:, :, 0] = 0  # Zero out contribution from red
    #blue = I[:,:,2]
    #I_green[:, :, 0] = 0  # Zero out contribution from red
    #I_green[:, :, 2] = 0  # Zero out contribution from blue

    red, green, blue = np.moveaxis(I, -1, 0)
    # Compute horizontal and vertical derivatives of red, blue and green channels through applying sobel filters
    r = filt.sobel_h(red)
    g = filt.sobel_h(green)
    b = filt.sobel_h(blue)
    u = array([r,g,b])
    r = filt.sobel_v(red)
    g = filt.sobel_v(green)
    b = filt.sobel_v(blue)
    v = array([r,g,b])
    gxx = color_dot_product(u, u)
    gyy = color_dot_product(v, v)
    gxy = color_dot_product(u, v)


    # Calculate gradient direction (direction of the largest colour change)
    theta = 0.5 * np.arctan(2 * gxy / (gxx - gyy))

    # Calculate the magnitude of the rate of change
    fTheta = np.sqrt(0.5 * ((gxx + gyy) + (gxx - gyy) * np.cos(2 * theta) + (2 * gxy * np.sin(2 * theta))))

    return np.array([theta, fTheta])


#Question 2
def test_blur_measure(I, min_sigma, max_sigma):

    #:param I: Image in float format
    #:param min_sigma: minimum standard deviation for gaussian filter mask that blurs image
    #:param max_sigma: maximum standard deviation for gaussian filter mask that blurs image
    #:return:

    s = min_sigma*max_sigma
    print(s)
    k_value = np.zeros(shape = [s, s])

    for sigma in range(min_sigma, max_sigma+1):
        gauss_I = filt.gaussian(I, sigma=sigma, multichannel=False)
        gradient_magnitude = color_sobel_edges(gauss_I)
        oned_arr = np.reshape(gradient_magnitude[0], gradient_magnitude.shape[1] * gradient_magnitude.shape[2])
        K = scipy.stats.kurtosis(oned_arr)
        sharpness = np.log(K + 3)
        s = sigma - min_sigma
        k_value[s,0] = sigma
        k_value[s, 1] = K

    return(k_value)

mushroom = io.imread('mushroom.jpg')
I = util.img_as_float(mushroom)
kurtosis = test_blur_measure(I, 1, 30)
plt.imshow(kurtosis)
plt.xlabel('Sigma')
plt.ylabel('Kurtosis')
plt.title('Kurtosis values for gaussian blurred gradient magnitudes with varying sigma values for mushroom.jpg')
plt.show()

#Question 3
def sharpness_map(I, window_size):
    '''

    Finish me!

    :param I: image in float format
    :param window_size: square window size (in pixels)
    :return: the array of local sharpness
    '''
    # compute local sharpness for each tiled,non-overlapping square window
    # Store local sharpness in array. Each entry represents one window of the input image.
    # Size of array = img_dimensions/window size
    shrp_arr = [I.shape[0]*I.shape[1]/window_size]
    for row in range(0,I.shape[0],11):
        #loop columns
        for col in range(0,I.shape[1],11):#enumerate(row):
            l = max(0,row-window_size)
            r = min(row+window_size,I.shape[0])
            u = max(0, col-window_size)
            d = min(row+window_size,I.shape[1])
            window = I[l:r,d:u]
            if (window.size != 0):
                gauss_I = filt.gaussian(window, sigma=15, multichannel=False)
                gradient_magnitude = color_sobel_edges(gauss_I)
                oned_arr = np.reshape(window, [gradient_magnitude[0] * gradient_magnitude[1]])
                K = scipy.stats.kurtosis(oned_arr)
                sharpness = np.log(K + 3)
                shrp_arr[row/window_size+col/window_size] = sharpness

mushroom = io.imread('mushroom.jpg')
I = util.img_as_float(mushroom)
sharpness = sharpness_map(I, 100)
plot(sharpness)
plt.colorbar()
plt.show()

#Question 4
water = io.imread('waterfall.jpg')
water_I = util.img_as_float(water)
kurtosis = test_blur_measure(I, 1, 30)
plot(kurtosis[:, 0], kurtosis[:, 1])
plt.xlabel('Sigma')
plt.ylabel('Kurtosis')
plt.title('Kurtosis values for gaussian blurred gradient magnitudes with varying sigma values for waterfall.jpg')
plt.show()

sharpness = sharpness_map(water_I, 100)
plot(sharpness)
plt.colorbar()
plt.show()
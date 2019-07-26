import skimage.io as io
import matplotlib.pyplot as plt
import skimage.util as util
import skimage.color as color

myimage = io.imread('asn0-files/jupiter_hs-2008-contrast-adjusted.png')
print(myimage.dtype)
float_img = util.img_as_float(myimage)
print(float_img.dtype)
myfigure = plt.figure()
plt.imshow(float_img)
plt.title('Float64 Image')
plt.show()
greyscale_img = color.rgb2gray(float_img)
plt.imshow(greyscale_img, cmap='gray', vmin=0, vmax=1)
plt.title('Float64 Image Greyscale')
plt.show()
plt.imshow(greyscale_img, cmap='gray', vmin=0, vmax=1)
plt.title('Float64 Image Greyscale Not Normalized')
plt.show()
import numpy as np
import skimage.io as io
import skimage.util as util
import skimage.measure as msr
import skimage.color as color
import matplotlib as plt
import glob
import skimage.restoration as restore
import os
import ast

#i.shape will give rows and columns, its a tuple(rows,columns)

#FUNTION VMF:
# Uses a vector median filter to denoise an image
#I: The image as a float
#radius: The radius that will be used to determine the neighborhood surrounding the center pixel
def VMF(I, radius):
    #loop all the rows of the image and save the index as well as the contentx
    for row in range(0,I.shape[0]):
        #loop columns
        for col in range(0,I.shape[1]):#enumerate(row):
            #The total amount of pixels in the square neighborhood
            nbhSz = ((radius*2)+1)**2
            #The pixels at the furthest boundaries for the neighborhood to the left, right, up, and down
            if (row > radius-1):
                uBnd = row - radius
            else:
                dBnd = row
            if (row + radius <= I.shape[0]):
                uBnd = row + radius
            else:
                uBnd = row
            if (col > radius - 1):
                lBnd = col - radius
            else:
                lBnd = col
            if (col + radius < I.shape[1]):
                rBnd = col + radius
            else:
                rBnd = col
            nbhd = I[lBnd:rBnd,dBnd:uBnd]   # array slice the neighborhood out of the image
            nbhd_rs = np.reshape(nbhd, [nbhd.shape[0]*nbhd.shape[1],3] )    #Turn the 3d array into a 2D array (3xn)
            Y = np.tile(nbhd_rs, (1,1,nbhSz))   #Make n planes of the 2d array in the line above
            nbhd_trps = np.transpose(np.expand_dims(nbhd_rs, axis=0), (2,1,0))  #create n planes of 1x3 arrays
            X = np.tile(nbhd_trps, (nbhSz, 1, 1))   #Give each plane n rows that are identical
            S = np.subtract(X,Y)
            D = np.abs(S)
            #k,temp = min(D[1:1:])
            #vk = Y[k,:,0]
            #I[rIdx,cIdx] = vk

#FUNCTION baseline:
#Makes graphs for PSNR and SSIM for noisy to noiseless images, and filtered to noiseless images
#param 1: a function that will filter the image
#param 2: The other arguments that param 1 requires
def baseline(filtFunc):
    images = [os.path.basename(x) for x in glob.glob('images/noisy/*.png')] #Save the names of all the images
    noiseList = list()  #Will store psnr and ssim values
    filtList = list()
    for image in images:
        #if the file is metaData
        if image == '.DS_Store':
            continue
        nsyImg = io.imread('images/noisy/'+image)
        nslsImg = io.imread('images/noiseless/'+image)
        nsyI = util.img_as_float(nsyImg)
        nslsI = util.img_as_float(nslsImg)
        #Call the filter function that was passed as a parameter, and pass the other arguments into this function
        #fltrPrm = (nsyI,*otherArgs)
        #print(fltrPrm)
        #filtImg = filtFunc(*fltrPrm)
        filtImg = restore.denoise_tv_chambolle(nsyI,weight=0.1,multichannel=True)
        #filtImg = exec(filtFunc)
        nSsim = msr.compare_ssim(nsyI, nslsI, multichannel=True)
        nPsnr = msr.compare_psnr(nslsI, nsyI)
        fSsim = msr.compare_ssim(filtImg, nslsI, multichannel=True)
        fPsnr = msr.compare_psnr(filtImg, nsyI)
        noiseList.append([image,nSsim,nPsnr])
        filtList.append([image,fSsim,fPsnr])
    nsySsimMean = noiseList[1].mean()
    filtSsimMean = filtList[1].mean()
    nsyPsnrMean = noiseList[2].mean()
    filtPsnrMean = filtList[2].mean()
    plt.bar(0.75, nsyPsnrMean, width=0.5)
    plt.bar(1.75, filtPsnrMean, width=0.5)
    plt.xticks([0.75, 1.75], ('PSNR Noisy', 'PSNR Filter'))
    plt.imshow()
    plt.bar(0.75, nsySsimMean, width=0.5)
    plt.bar(1.75, filtSsimMean, width=0.5)
    plt.xticks([0.75, 1.75], ('SSIM Noisy', 'SSIM Filter'))
    plt.imshow()
    plt.savefi
    #plt.show()

radius = 3
myimage = io.imread('images/noisy-test/12003.png')
I = util.img_as_float(myimage)
VMF(I,radius)
#baseline('restore.denoise_tv_chambolle(nsyI,weight=0.1,multichannel=True)')

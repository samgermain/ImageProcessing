import numpy as np
import skimage.io as io
import skimage.util as util
import skimage.color as color
import matplotlib as plt

#u = np.array([[1,2,3],[4,5,6]])
#print(u)
#v = np.tile(u, (2,1))
#w = np.transpose(np.expand_dims(u, axis=3), (2,1,0))
#print(w)

#i.shape will give rows and columns, its a tuple(rows,columns)
def L1_Norm(I, radius):
    nbhSz = ((radius*2)+1)**2
    lBnd = rIdx - radius
    rBnd = rIdx + radius
    uBnd = cIdx - radius
    dBnd = cIdx + radius
    nbhd = I[lBnd:rBnd,dBnd:uBnd]
    nbhd_rs = np.reshape(nbhd, [nbhd.shape[0]*nbhd.shape[1],3] )    #Some of them are blank
    Y = np.tile(nbhd_rs, (1,1,nbhSz))
    nbhd_trps = np.transpose(np.expand_dims(nbhd_rs, axis=0), (2,1,0))
    X = np.tile(nbhd_trps, (nbhSz, 1, 1))
    D = abs(X-Y)
    k,temp = min(D[1:1:])
    vk = Y[k,:,0]
    return vk

radius = 3
myimage = io.imread('images/noisy-test/12003.png')
I = util.img_as_float(myimage)
for rIdx, row in enumerate(myimage):
    for cIdx, col in enumerate(row):
        vk = L1_Norm(I, 3)
        i[rIdx,cIdx] = vk

plt.imshow(I)
plt.show()
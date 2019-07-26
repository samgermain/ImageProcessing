import skimage.feature as feat
import skimage.util as util
import glob
import skimage.io as io
import numpy as np
import os

trn_images = [os.path.basename(x) for x in glob.glob('brodatztraining/*.png')]
trn_images.sort()
trn_feats = []
lbp_trn = np.array([])
for i in range(0, len(trn_images)):
    B = util.img_as_ubyte(io.imread('brodatztraining/' + trn_images[i]))
    gclms = feat.greycomatrix(B, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], normed=True)
    #stats = []
    #stats.append(feat.greycoprops(gclms, prop='energy'))
    #stats.append(feat.greycoprops(gclms, prop='contrast'))
    #stats.append(feat.greycoprops(gclms, prop='correlation'))
    #stats.append(feat.greycoprops(gclms, prop='homogeneity'))
    #trn_feats.append(stats)
    trn_feats.append(feat.greycoprops(gclms, prop='energy'))
    #trn_feats.append(feat.greycoprops(gclms, prop='contrast'))
    lbp1 = feat.local_binary_pattern(B, 8, 1, method='uniform')
    lbp2 = feat.local_binary_pattern(B, 8, 1, method='var')
    lbp3 = np.concatenate((lbp1, lbp2))
    hist, other_stuff = np.histogram(lbp1, bins=26, range=(0, 7000))
    lbp_trn = np.concatenate((lbp_trn,hist))
    #lbp3 = np.concatenate(lbp1,lbp2)
    #lbp_trn.append(lbp3)
#print(trn_feats.shape)
trn_feats = np.transpose(np.asarray(trn_feats), (0,2,1))[:,:,0]
#print(trn_feats.shape)
lbp_trn = lbp_trn.reshape(-1, 1)

#2
tst_images = [os.path.basename(x) for x in glob.glob('brodatztesting/*.png')]
tst_images.sort()
tst_feats = []
lbp_tst = np.array([])
for i in range(0, len(tst_images)):
    B = util.img_as_ubyte(io.imread('brodatztesting/' + tst_images[i]))
    gclms = feat.greycomatrix(B, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], normed=True)
    #stats = []
    #stats.append(feat.greycoprops(gclms, prop='energy'))
    #stats.append(feat.greycoprops(gclms, prop='contrast'))
    #stats.append(feat.greycoprops(gclms, prop='correlation'))
    #stats.append(feat.greycoprops(gclms, prop='homogeneity'))
    #tst_feats.append(stats)
    tst_feats.append(feat.greycoprops(gclms, prop='energy'))
    lbp1 = feat.local_binary_pattern(B, 8, 1, method='uniform')
    lbp2 = feat.local_binary_pattern(B, 8, 1, method='var')
    lbp3 = np.vstack((lbp1, lbp2)).T
    hist, other_stuff = np.histogram(lbp1, bins=26, range=(0, 7000))
    lbp_tst = np.concatenate((lbp_tst, hist))
tst_feats = np.transpose(np.asarray(tst_feats), (0, 2, 1))[:, :, 0]
#tst_feats = np.asarray(tst_feats)

#3
trn_labels = np.zeros(len(trn_images))
clazz = 1
for i in range(0,len(trn_images)):
    if (i%15 == 0):
        clazz += 1
    trn_labels[i] = clazz

tst_labels = np.zeros(len(tst_images))
clazz = 1
for i in range(0,len(tst_images)):
    if (i%40 == 0):
        clazz += 1
    tst_labels[i] = clazz

#4
import sklearn.neighbors as neighbors
neighbs = 3
knn = neighbors.KNeighborsClassifier(n_neighbors=neighbs)
lnn = neighbors.KNeighborsClassifier(n_neighbors=neighbs)
#print("trn_feats: ", trn_feats.shape,"trn_labels: ", trn_labels.shape)
knn.fit(trn_feats, trn_labels)
lnn.fit(trn_feats, trn_labels)

#5
predict_labels = knn.predict(tst_feats)
lpredict_labels = lnn.predict(tst_feats)

#
confusion = np.zeros((clazz,clazz))
for i in range(0, len(tst_labels)):
    lt = int(tst_labels[i])
    l = int(predict_labels[i])
    confusion[lt-1, l-1] += 1

print("Confusion Matrix for graycomatrix\n", confusion)

print("Classification rate for graycomatrix: ", np.trace(confusion)/np.sum(confusion) * 100)

confusion = np.zeros((clazz,clazz))
for i in range(0, len(tst_labels)):
    lt = int(tst_labels[i])
    l = int(lpredict_labels[i])
    confusion[lt-1, l-1] += 1

print("Confusion Matrix for local binary pattern\n", confusion)

print("Classification rate for local binary pattern: ", np.trace(confusion)/np.sum(confusion) * 100)
import os
def myPrint(text, path, consolePrint=True):
    if not os.path.exists(path+'/reports/'):
        os.mkdir(path+'/reports/')
    if consolePrint:
        print(text)
    print(text, file=open(path+'/reports/output.txt', 'a'))
    
def myLog(text, path):
    myPrint(text, path, consolePrint=False)
  
def visualizeDataset(dataset, plotSize=[4,4]):
    import matplotlib.pyplot as plt
    plt.figure()
    for num in range(len(dataset)):
        plt.subplot(plotSize[0],plotSize[1],num+1)
        centerSlice = int(dataset.shape[2]/2)
        if len(dataset.shape) == 5:
            plt.imshow(dataset[num, :, centerSlice, :, 0], cmap='gray')
        else:
            plt.imshow(dataset[num, :, centerSlice, :], cmap='gray')
        plt.axis('off')
    plt.suptitle('Center Coronal Slice\nfrom each training image')
    
import re
def sortHuman(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9])', key)]
    l.sort(key=alphanum)
    return l

import numpy as np
from skimage.measure import label
def getLargestCC(segmentation):
    labels = label(segmentation) # Convert each connected component to a label
    cnt = np.bincount(labels.flat) # Count each label
    cnt[0] = 0 # Background label count set to zero
    largestCC = labels == np.argmax(cnt) # Get the largest label
    largestCC = largestCC.astype(int) # Convert bool to int image
    return largestCC
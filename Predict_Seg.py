# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:19:55 2018

@author: Saeed Mhq
"""

# In the name of GOD

from Models import UNet_3D

run = 'run2'
resultsPath = './Results/' + run
    
def getTestIndices(path):
    text_file = open(path, "r")
    files = text_file.readlines()
    ids = []
    for i in range(len(files)):
        files[i] = os.path.basename(files[i].split('\t')[1])
        s=files[i].split('_')
        ids.append([int(s) for s in s[-1].split('-') if s.isdigit()][0])
    return files, ids

'''--------------Load Data--------------'''

from Utils.load_dataset import prepare_dataset

listPath = resultsPath + '/reports/valid_list.txt' 
testFiles, ids = getTestIndices(listPath)

datasetDir = './Dataset/'
images, masks, _, _ = prepare_dataset(datasetDir, split=1., scaleFactor=0.5)
testImages = images[ids]

'''--------------Build Model--------------'''
img_size = images.shape[1:]
model = UNet_3D.UNet_3D(img_size)
weightsPath =resultsPath + '/weights/UNet_3D_model.hdf5'
model.load_weights(weightsPath)

'''--------------Prediction---------------'''
import numpy as np
import os
import SimpleITK as sitk

predicted = model.predict(testImages)
#predicted = np.flipud(predicted)
predDir = resultsPath + '/Predicted/'
if not os.path.exists(predDir):
    os.mkdir(predDir)
for i in range(len(predicted)):
    volOut = sitk.GetImageFromArray(predicted[i,:,:,:,0])
    outFile = os.path.join(predDir, testFiles[i][:-8]+'_pred.nii.gz')
    sitk.WriteImage(volOut, outFile)
    print('...save ' + outFile)
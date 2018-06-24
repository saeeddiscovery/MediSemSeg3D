# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:19:55 2018

@author: Saeed Mhq
"""

# In the name of GOD

from Models import UNet_3D, DilatedNet_3D
import os


run = 'run2'
resultsPath = './Results/' + run
    
def getIndicesFromFile(path):
    text_file = open(path, "r")
    files = text_file.readlines()
    ids = []
    for i in range(len(files)):
        files[i] = os.path.basename(files[i].split('\t')[1])
        s=files[i].split('_')
        imgID = [int(s) for s in s[-1].split('-') if s.isdigit()][0]
        ids.append(imgID-1)
    return files, ids

'''--------------Load Data--------------'''

#from Utils.load_dataset import prepare_dataset
from Utils.load_dataset import load_list

listPath = resultsPath + '/reports/valid_list_images.txt' 
testImages, testFiles = load_list(listPath)

#testFiles, ids = getIndicesFromFile(listPath)
#datasetDir = './Dataset/'
#images, masks, _, _ = prepare_dataset(datasetDir, split=1., scaleFactor=0.5)
#testImages = images[ids]


'''--------------Build Model--------------'''
modelName = 'UNet'
#modelName = 'DilatedNet'
img_size = testImages.shape[1:]
if modelName == 'UNet':
    model = UNet_3D.UNet_3D(img_size)
elif modelName == 'DilatedNet':
    model = DilatedNet_3D.DilatedNet_3D(img_size)

weightsPath =resultsPath + '/weights/' + modelName + '_3D_model.hdf5'
model.load_weights(weightsPath)

'''--------------Prediction---------------'''
import numpy as np
import SimpleITK as sitk

print('------------<  Dataset Info >------------')
print('Model: ' + modelName)
predDir = resultsPath + '/Predicted/'
if not os.path.exists(predDir):
    os.mkdir(predDir)
predicted = []
threshold = 0.7
for i,img in enumerate(testImages):
    img = img[np.newaxis,:]
    print('... Predicting ' + testFiles[i])
    predicted = np.squeeze(model.predict(img))
    predicted = predicted[::-1]
    predicted[predicted <= threshold] = 0
    predicted[predicted > threshold] = 1
    volOut = sitk.GetImageFromArray(predicted)
    outFile = os.path.join(predDir, testFiles[i][:-7]+'_pred.nii.gz')
    sitk.WriteImage(volOut, outFile)
    print('saved as ' + outFile)
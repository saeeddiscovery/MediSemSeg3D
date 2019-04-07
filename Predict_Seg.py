# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:19:55 2018

@author: Saeed Mhq
"""

# In the name of GOD

from Models import UNet_3D, DilatedNet_3D
import os
import tensorflow as tf
import time
from Utils.utils import getLargestCC

# GPU Memory Management
K = tf.keras.backend
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
K.set_session(sess)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    

run = 'run4'
resultsPath = './Results/' + run
hybridModel = True
modelName = 'UNet'
#modelName = 'DilatedNet'
dsmType = 'CAE'
#dsmType = 'CVAE'


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

from Utils.load_dataset import prepare_dataset, prepare_test
from Utils.load_dataset import load_list

listPath = resultsPath + '/reports/valid_list_images.txt' 


#testFiles, ids = getIndicesFromFile(listPath)
datasetDir = './Dataset/Dataset_MICCAI2007_Preprocessed_eval/'
#datasetDir = './Dataset/MICCAI_test_Preprocessed/'
#images, masks, _, _ = prepare_dataset(datasetDir, split=1., scaleFactor=0.5)
#testImages = images[ids]

#testImages, testFiles = load_list(listPath, savelist=False)
#testImages, testFiles = prepare_test(datasetDir, scaleFactor=0.25)
testImages, testFiles = prepare_test(datasetDir, shuffle=False, isTest=False)
img_size = testImages.shape[1:] 
#img_size = (128, 128, 128, 1) 

'''--------------Build Model--------------'''
if modelName == 'UNet':
    segModel = UNet_3D.UNet_3D(img_size)
elif modelName == 'DilatedNet':
    segModel = DilatedNet_3D.DilatedNet_3D(img_size)
        
if hybridModel:      
    latent_dim = 64
    batch_size = 1
    if dsmType == 'CAE':
        from Deep3DSM.Models import CAE_3D
        dsmModel = CAE_3D.FullModel(img_size, latent_dim)
        encoder = CAE_3D.get_encoder_from_CAE3D(dsmModel)
    elif dsmType == 'CVAE':
        from Deep3DSM.Models import CVAE_3D
        encoder,_, dsmModel = CVAE_3D.CVAE(img_size, batch_size, latent_dim)

    inLayer = tf.keras.layers.Input(shape=img_size)
#    if mask_image:
#        outSeg = tf.keras.layers.dot([inLayer, segModel(inLayer)], axes=-1, normalize=True)
#        model = tf.keras.Model(inLayer, [segModel(inLayer), encoder(outSeg)])
#    else:
    model = tf.keras.Model(inLayer, [segModel(inLayer), encoder(segModel(inLayer))])
    weightsPath =resultsPath + '/weights/' + modelName + '_' + dsmType + '_model_v.hdf5'       

else:        
    model = segModel
    weightsPath =resultsPath + '/weights/' + modelName + '_model_v.hdf5'

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
threshold = 0.1

if (hybridModel == True):
    pred0 = np.zeros_like(testImages)   
    pred1 = np.zeros([len(testImages), latent_dim], dtype='float32')
    for i,img in enumerate(testImages):
        img = img[np.newaxis,:]
        print('... Predicting image {} of {}'.format(i+1, len(testImages)))
        start = time.time()
        pred0[i] = np.squeeze(model.predict(img)[0])[:,:,:,np.newaxis]
        dur = time.time() - start
        print("duration: " , dur)
        pred1[i] = np.squeeze(model.predict(img)[1])
        predicted = pred0[i][::-1]
        predicted[predicted <= threshold] = 0
        predicted[predicted > threshold] = 1
        predicted = np.squeeze(predicted)
        predicted = getLargestCC(predicted)
        volOut = sitk.GetImageFromArray(predicted)
        outFile = os.path.join(predDir, os.path.basename(testFiles[i])[:-7]+'_pred.nii.gz')
        sitk.WriteImage(volOut, outFile)
        print('saved as ' + outFile)
    outFile = os.path.join(predDir, 'latent')
    np.savez_compressed(outFile, pred1)
    print('saved as ' + outFile)

else:
    for i,img in enumerate(testImages):
        img = img[np.newaxis,:]
        print('... Predicting image {} of {}'.format(i+1, len(testImages)))
        predicted = np.squeeze(model.predict_on_batch(img))
        predicted = predicted[::-1]
        predicted[predicted <= threshold] = 0
        predicted[predicted > threshold] = 1
        predicted = getLargestCC(predicted)
        predicted = np.squeeze(predicted)
        volOut = sitk.GetImageFromArray(predicted)
        outFile = os.path.join(predDir, os.path.basename(testFiles[i])[:-7]+'_pred.nii.gz')
        sitk.WriteImage(volOut, outFile)
        print('saved as ' + outFile)
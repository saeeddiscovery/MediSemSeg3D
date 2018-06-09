# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:33:57 2018

@author: Saeed Mhq
"""

# In the name of GOD

import os, glob

modelName = 'UNet'
#modelName = 'VNet'

import re
def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9])', key)]
    l.sort(key=alphanum)
    return l

resultsDir = './Results/'
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)
if os.listdir(resultsDir)==[]:
    currRun = '/run1'
    os.mkdir(resultsDir+currRun)
else:
    runs = glob.glob(os.path.join(resultsDir, 'run*'))
    runs = sort_human(runs)
    currRun = '/run' + str(int(runs[-1][13:])+1)
    os.mkdir(resultsDir+currRun)

'''--------------Load Data--------------'''

from Utils.load_dataset import prepare_dataset
 
datasetDir = './Dataset/'
dTrain, mTrain, dValid, mValid = prepare_dataset(datasetDir, logPath=resultsDir+currRun, scaleFactor=0.5)

#-------Visualize Dataset-------#
#from Utils.utils import visualizeDataset
#visualizeDataset(dTrain, plotSize=[4,4])
#visualizeDataset(mTrain, plotSize=[4,4])

'''--------------Build Model--------------'''

import tensorflow as tf
from Models import UNet_3D
from Utils.utils import myPrint
import numpy as np
import datetime
K = tf.keras.backend
    
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def summary(model): # Compute number of params in a model (the actual number of floats)
    trainParams = sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
    myPrint('------------< Model Summary >------------', path=resultsDir+currRun)
    myPrint('...Total params:      {:,}'.format(model.count_params()), path=resultsDir+currRun)
    myPrint('...Trainable params:  {:,}'.format(trainParams), path=resultsDir+currRun)

img_size = dTrain.shape[1:]

if modelName == 'UNet':
    model = UNet_3D.UNet_3D(img_size)
    
model.compile(optimizer=tf.keras.optimizers.Adam(),
               loss=dice_coef_loss, metrics=['accuracy'])
summary(model)
tf.keras.utils.plot_model(model, to_file=resultsDir+currRun+'/reports/' + modelName + '_Model.png', show_shapes=True)


'''--------------Train Model--------------'''
myPrint('------------< Start Training >-----------', path=resultsDir+currRun)
start = datetime.datetime.now()
myPrint('Start: {}'.format(start.ctime()[:-5]), path=resultsDir+currRun)

batch_size = 1
epochs = 100

weightsDir = resultsDir+currRun+'/weights'
if not os.path.exists(weightsDir):
    os.mkdir(weightsDir)
    
if modelName == 'UNet':
    #model_file = "UNet_3D_model-{epoch:02d}-{val_loss:.2f}.hdf5"
    model_file = weightsDir+"/UNet_3D_model.hdf5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file,
                                                          monitor='loss',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          save_weights_only=True)
    logger = tf.keras.callbacks.CSVLogger(resultsDir+currRun+'/reports/training.log', separator='\t')
    tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/UNet'+currRun)
    callbacks = [tensorBoard, model_checkpoint, logger]

model.fit(dTrain, mTrain, shuffle=True, epochs=epochs, batch_size=batch_size,
          validation_data=(dValid, mValid), callbacks=callbacks)


end = datetime.datetime.now()
elapsed = end-start
myPrint('Start: {}'.format(start.ctime()[:-5]), path=resultsDir+currRun)
myPrint('Train time: {}'.format(elapsed), path=resultsDir+currRun)
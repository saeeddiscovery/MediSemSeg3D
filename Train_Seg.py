# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:33:57 2018

@author: Saeed Mhq
"""

# In the name of GOD

import os, glob

#modelName = 'UNet'
#modelName = 'DilatedNet'
modelName = 'UNet+DSM'
#dsmType = 'CAE'
dsmType = 'CVAE'

from Utils.utils import sortHuman
resultsDir = './Results/'
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)
if os.listdir(resultsDir)==[]:
    currRun = '/run1'
    os.mkdir(resultsDir+currRun)
else:
    runs = glob.glob(os.path.join(resultsDir, 'run*'))
    runs = sortHuman(runs)
    currRun = '/run' + str(int(runs[-1][13:])+1)
    os.mkdir(resultsDir+currRun)

'''--------------Load Data--------------'''

from Utils.load_dataset import prepare_dataset
 
datasetDir = './Dataset/'
dTrain, mTrain, dValid, mValid = prepare_dataset(datasetDir, logPath=resultsDir+currRun, scaleFactor=1)

#-------Visualize Dataset-------#
#from Utils.utils import visualizeDataset
#visualizeDataset(dValid, plotSize=[10,11])
#visualizeDataset(mValid, plotSize=[10,11])

'''--------------Build Model--------------'''

import tensorflow as tf
from Utils.utils import myPrint, myLog
import numpy as np
import datetime
K = tf.keras.backend

# GPU Memory Management
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
K.set_session(sess)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

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

#img_size = dTrain.shape[1:]
img_size = (128, 128, 128, 1)
batch_size = 1
myPrint('...Input image size: {}'.format(img_size), path=resultsDir+currRun)
myPrint('...Batch size: {}'.format(batch_size), path=resultsDir+currRun)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

lr = 0.01
decay = 1e-2

if modelName == 'UNet':
    from Models import UNet_3D
    model = UNet_3D.UNet_3D(img_size) 
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                   loss=dice_coef_loss, metrics=['accuracy'],
                   options=run_opts)
if modelName == 'DilatedNet':
    from Models import DilatedNet_3D
    model = DilatedNet_3D.DilatedNet_3D(img_size) 
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                   loss=dice_coef_loss, metrics=['accuracy'],
                   options=run_opts)
elif modelName == 'UNet+DSM':
    from Models import UNet_3D
    latent_dim = 128
    segModel = UNet_3D.UNet_3D(img_size)
    inLayer = tf.keras.layers.Input(shape=img_size)
    if dsmType == 'CAE':
        from Models import CAE_3D
        dsmWeightsPath = r'./Deep3DSM/CAE/run-1_b1/CAE_3D_encoder.hdf5'
        dsmModel = CAE_3D.FullModel(img_size, latent_dim)
#        dsmModel.load_weights(dsmWeightsPath)
#        dsmModel.trainable = False 
        encoder = CAE_3D.get_encoder_from_CAE3D(dsmModel)
        encoder.load_weights(dsmWeightsPath)
        encoder.trainable = False
        tf.keras.utils.plot_model(encoder, to_file=resultsDir+currRun+'/reports/' + modelName + '_Encoder.png', show_shapes=True)
#        model = tf.keras.Model(inLayer, encoder(segModel(inLayer)))
        model = tf.keras.Model(inLayer, [segModel(inLayer), encoder(segModel(inLayer))])
#        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
#                       loss=tf.keras.losses.binary_crossentropy,
#                       metrics=['accuracy'], options=run_opts)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                       loss=[dice_coef_loss, tf.keras.losses.binary_crossentropy],
                       metrics=['accuracy'], options=run_opts)
    elif dsmType == 'CVAE':
        from Models import CVAE_3D
        dsmWeightsPath = r'./Deep3DSM/CVAE/run-1_b1/CVAE_3D_encoder.hdf5'
        encoder,_, dsmModel = CVAE_3D.CVAE(img_size, batch_size, latent_dim)
#        dsmModel.load_weights(dsmWeightsPath)
#        dsmModel.trainable = False 
        encoder.load_weights(dsmWeightsPath)
        encoder.trainable = False
        tf.keras.utils.plot_model(encoder, to_file=resultsDir+currRun+'/reports/' + modelName + '_Encoder.png', show_shapes=True)
#        model = tf.keras.Model(inLayer, encoder(segModel(inLayer)))
        model = tf.keras.Model(inLayer, [segModel(inLayer), encoder(segModel(inLayer))])
#        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
#                       loss=tf.keras.losses.binary_crossentropy,
#                       metrics=['accuracy'], options=run_opts)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                       loss=[dice_coef_loss, tf.keras.losses.kullback_leibler_divergence],
                       metrics=['accuracy'], options=run_opts)
    
summary(model)
tf.keras.utils.plot_model(model, to_file=resultsDir+currRun+'/reports/' + modelName + '_Model.png', show_shapes=True)


'''--------------Train Model--------------'''
myPrint('------------< Start Training >-----------', path=resultsDir+currRun)
start = datetime.datetime.now()
myPrint('...Start: {}'.format(start.ctime()[:-5]), path=resultsDir+currRun)

epochs = 150

weightsDir = resultsDir+currRun+'/weights'
if not os.path.exists(weightsDir):
    os.mkdir(weightsDir)

myLog('epoch\tlr\tloss\tval_loss', path=resultsDir+currRun)
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self,epoch, logs={}):
        # Things done on beginning of epoch. 
        return
    def on_epoch_end(self, epoch, logs={}):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
#        iterations = self.model.optimizer.iterations
#        lr_with_decay = lr / (1. + decay * (epoch)//2))
        lr_with_decay = lr / (1. + decay * epoch)
        myLog(str(epoch) +'\t' + str(K.eval(lr_with_decay)) +'\t' + str(logs.get("loss")) +'\t' + str(logs.get("val_loss")), path=resultsDir+currRun)
        
if modelName == 'UNet' or modelName == 'DilatedNet':
    model_file = weightsDir+"/" + modelName + "_3D_model-{epoch:02d}-{val_loss:.2f}.hdf5"
#    model_file = weightsDir+"/" + modelName + "_3D_model.hdf5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file,
                                                          monitor='val_loss',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          save_weights_only=True)
    logger = tf.keras.callbacks.CSVLogger(resultsDir+currRun+'/reports/training.log', separator='\t')
    tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/'+modelName+currRun)
    lrs = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr / (1. + decay * epoch))
    callbacks = [tensorBoard, model_checkpoint, logger, MyCallback(), lrs]

    model.fit(dTrain, mTrain, shuffle=True, epochs=epochs, batch_size=batch_size,
              validation_data=(dValid, mValid), callbacks=callbacks)
    
elif modelName == 'UNet+DSM':
    mTrain_latent = np.zeros([len(mTrain), 128], dtype='float32')
    for i,img in enumerate(mTrain):
        img = img[np.newaxis,:]
        print('... Predicting image ' + str(i))
        mTrain_latent[i] = np.squeeze(encoder.predict(img))
#        predicted = predicted[::-1]
    mValid_latent = np.zeros([len(mValid), 128], dtype='float32')  
    for i,img in enumerate(mValid):
        img = img[np.newaxis,:]
        print('... Predicting image ' + str(i))
        mValid_latent[i] = np.squeeze(encoder.predict(img))
#        predicted = predicted[::-1]
        
#    pred0 = np.zeros_like(mValid)   
#    pred1 = np.zeros([len(mValid), 128], dtype='float32')
#    for i,img in enumerate(mValid):
#        img = img[np.newaxis,:]
#        print('... Predicting image ' + str(i))
#        pred0[i] = np.squeeze(model.predict(img)[0])[:,:,:,np.newaxis]
#        pred1[i] = np.squeeze(model.predict(img)[1])

    #model_file = "UNet_3D_model-{epoch:02d}-{val_loss:.2f}.hdf5"
    model_file = weightsDir+"/UNet+DSM_" + dsmType + "_model.hdf5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file,
                                                          monitor='loss',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          save_weights_only=True)
    logger = tf.keras.callbacks.CSVLogger(resultsDir+currRun+'/reports/training.log', separator='\t')
    tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/UNet+DSM'+currRun)
    lrs = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr / (1. + decay * epoch))
    callbacks = [tensorBoard, model_checkpoint, logger, MyCallback(), lrs]
    if dsmType == 'CAE':
    #    model.fit(dTrain, mTrain_latent, shuffle=True, epochs=epochs, batch_size=batch_size,
    #              validation_data=(dValid, mValid_latent), callbacks=callbacks)
        model.fit(dTrain, [mTrain, mTrain_latent], shuffle=True, epochs=epochs, batch_size=batch_size,
                  validation_data=(dValid, [mValid, mValid_latent]), callbacks=callbacks)   
    elif dsmType == 'CVAE':
    #    model.fit(dTrain, mTrain_latent, shuffle=True, epochs=epochs, batch_size=batch_size,
    #              validation_data=(dValid, mValid_latent), callbacks=callbacks)
        model.fit(dTrain, [mTrain, mTrain_latent], shuffle=True, epochs=epochs, batch_size=batch_size,
                  validation_data=(dValid, [mValid, mValid_latent]), callbacks=callbacks)  


end = datetime.datetime.now()
elapsed = end-start
myPrint('...End: {}'.format(end.ctime()[:-5]), path=resultsDir+currRun)
myPrint('...Train time: {}'.format(elapsed), path=resultsDir+currRun)
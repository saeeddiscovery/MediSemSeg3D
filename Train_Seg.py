# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:33:57 2018

@author: Saeed Mhq
"""

# In the name of GOD

import tensorflow as tf
import os, glob
K = tf.keras.backend

## GPU Memory Management
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
##config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#sess = tf.Session(config=config)
#K.set_session(sess)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

hybridModel = True
modelName = 'UNet'
#modelName = 'DilatedNet'
#modelName = 'DilatedNet2'
dsmType = 'CAE'
#dsmType = 'CVAE'
mask_image = False

from Utils.utils import sortHuman
resultsDir = './Results/'
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)

runs = glob.glob(os.path.join(resultsDir, 'run*'))
runs = sortHuman(runs)
if len(runs) == 0:
    currRun = '/run1'
    os.mkdir(resultsDir+currRun)
else:
    currRun = '/run' + str(int(runs[-1][13:])+1)
    os.mkdir(resultsDir+currRun)

import numpy as np
def maskImages(images, masks):
    maskedImgs = np.zeros_like(masks)
    for i in range(len(maskedImgs)):
        image = images[i,:,:,:,0]
        mask = masks[i,:,:,:,0]
        maskedImgs[i,:,:,:,0] = image * mask
    return maskedImgs
'''------------------------------- Load Data -------------------------------'''

from Utils.load_dataset import prepare_dataset, load_list
 
datasetDir = './Dataset/'
#dTrain, mTrain, dValid, mValid = prepare_dataset(datasetDir, logPath=resultsDir+currRun, scaleFactor=1)

train_images = './imageLists/train_list_images.txt' 
train_masks = './imageLists/train_list_masks.txt'
valid_images = './imageLists/valid_list_images.txt' 
valid_masks = './imageLists/valid_list_masks.txt'
dTrain, _ = load_list(train_images, logPath=resultsDir+currRun)
mTrain, _ = load_list(train_masks, logPath=resultsDir+currRun)
dValid, _ = load_list(valid_images, logPath=resultsDir+currRun)
mValid, _ = load_list(valid_masks, logPath=resultsDir+currRun)

if mask_image:
    mTrain_masked = maskImages(dTrain, mTrain)
    mValid_masked = maskImages(dValid, mValid)
    
##-------Visualize Dataset-------#
#from Utils.utils import visualizeDataset
#visualizeDataset(dTrain, plotSize=[5,6])
#visualizeDataset(mTrain, plotSize=[5,6])

'''------------------------------ Build Model ------------------------------'''

from Utils.utils import myPrint, myLog
import datetime

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def myLoss(y_true, y_pred):
    a = 0.8
    BCE = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    DCE = -dice_coef(y_true, y_pred)
    myLoss = a*DCE + (1-a)*BCE
    return myLoss

def summary(model, modelType): # Compute number of params in a model (the actual number of floats)
    trainParams = sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
    myPrint('-------------< {} Model >--------------'.format(modelType), path=resultsDir+currRun)
    myPrint('...Total params:      {:,}'.format(model.count_params()), path=resultsDir+currRun)
    myPrint('...Trainable params:  {:,}'.format(trainParams), path=resultsDir+currRun)

img_size = dTrain.shape[1:]
#img_size = (None, None, None, 1)
#img_size = (128, 128, 128, 1)

batch_size = 3
myPrint('...Input image size: {}'.format(img_size), path=resultsDir+currRun)
myPrint('...Batch size: {}'.format(batch_size), path=resultsDir+currRun)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

lr = 0.0005
decay = 1e-5

if modelName == 'UNet':
    from Models import UNet_3D
    segModel = UNet_3D.UNet_3D(img_size) 
elif modelName == 'DilatedNet':
    from Models import DilatedNet_3D
    segModel = DilatedNet_3D.DilatedNet_3D(img_size) 
elif modelName == 'DilatedNet2':
    from Models import DilatedNet_3D
    segModel = DilatedNet_3D.DilatedNet_3D_2(img_size) 
        
if not hybridModel:
    model = segModel
#    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
#                   loss=dice_coef_loss, metrics=[dice_coef],
#                   options=run_opts)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                   loss=dice_coef_loss, metrics=[dice_coef])
    myPrint('...Loss: Dice', path=resultsDir+currRun)
#    myPrint('...Loss: {}*Dice + {}*BCE'.format(0.8, 1-0.8), path=resultsDir+currRun)
    
elif hybridModel:      
    latent_dim = 64
    myPrint('...Latent dim: {}'.format(latent_dim), path=resultsDir+currRun)
    if dsmType == 'CAE':
        from Deep3DSM.Models import CAE_3D
        dsmWeightsPath = r'./MediSemSeg3D_results/Deep3DSM/CAE/run-2-64-noisy/weights/CAE_3D_encoder.hdf5'
        dsmModel = CAE_3D.FullModel(img_size, latent_dim)
        encoder = CAE_3D.get_encoder_from_CAE3D(dsmModel)
    elif dsmType == 'CVAE':
        from Deep3DSM.Models import CVAE_3D
        dsmWeightsPath = r'./MediSemSeg3D_results/Deep3DSM/CVAE/run-2-64-noisy/weights/CVAE_3D_encoder.hdf5'
        encoder,_, dsmModel = CVAE_3D.CVAE(img_size, batch_size, latent_dim)
        
    encoder.load_weights(dsmWeightsPath)
    encoder.trainable = False
    alpha = 0.8
    inLayer = tf.keras.layers.Input(shape=img_size)
    if mask_image:
        outSeg = tf.keras.layers.dot([inLayer, segModel(inLayer)], axes=-1, normalize=True)
        model = tf.keras.Model(inLayer, [segModel(inLayer), encoder(outSeg)])
    else:
        model = tf.keras.Model(inLayer, [segModel(inLayer), encoder(segModel(inLayer))])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                   loss=[dice_coef_loss, tf.keras.losses.binary_crossentropy],
                   loss_weights = [alpha, 1-alpha],
                   metrics=[dice_coef])
#    model = tf.keras.Model(inLayer, encoder(segModel(inLayer)))
#    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
#                   loss=tf.keras.losses.binary_crossentropy,
#                   metrics=['accuracy'], options=run_opts)

    myPrint('...Loss: {}*Dice + {}*BCE'.format(alpha, 1-alpha), path=resultsDir+currRun)
        
    tf.keras.utils.plot_model(encoder, to_file=resultsDir+currRun+'/reports/' + modelName + '_Encoder.png', show_shapes=True)
    summary(encoder, modelType='Encoder')
    
summary(model, modelType='Full')
tf.keras.utils.plot_model(model, to_file=resultsDir+currRun+'/reports/' + modelName + '_Model.png', show_shapes=True)

'''------------------------------ Callbacks --------------------------------'''

weightsDir = resultsDir+currRun+'/weights/'
if not os.path.exists(weightsDir):
    os.mkdir(weightsDir)
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self,epoch, logs={}):
        # Things done on beginning of epoch. 
        return
    def on_epoch_end(self, epoch, logs={}):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
#        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * epoch)
        myLog(str(epoch) +'\t' + str(K.eval(lr_with_decay)) +'\t' + str(logs.get("loss")) +'\t' + str(logs.get("val_loss")), path=resultsDir+currRun)
modelNameFull =  modelName + '_' + dsmType if hybridModel else modelName      
#model_file = weightsDir+"/" + modelNameFull + "_3D_model-{epoch:02d}-{val_loss:.2f}.hdf5"
model_file_v = weightsDir + modelNameFull + "_model_v.hdf5"
model_file_t = weightsDir + modelNameFull + "_model_t.hdf5"
model_checkpoint_v = tf.keras.callbacks.ModelCheckpoint(model_file_v,
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True)
model_checkpoint_t = tf.keras.callbacks.ModelCheckpoint(model_file_t,
                                                      monitor='loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True)
logger = tf.keras.callbacks.CSVLogger(resultsDir+currRun+'/reports/training.log', separator='\t')
tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/'+modelName+currRun)
lrs = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr / (1. + decay * epoch))
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau()
#callbacks = [tensorBoard, model_checkpoint_v, model_checkpoint_t, logger, MyCallback(), lrs, ReduceLROnPlateau]
callbacks = [tensorBoard, model_checkpoint_v, model_checkpoint_t, logger, ReduceLROnPlateau]

'''-------------------------------Train Model-------------------------------'''

myPrint('--------------< Training >---------------', path=resultsDir+currRun)
start = datetime.datetime.now()
myPrint('...Start: {}'.format(start.ctime()[:-5]), path=resultsDir+currRun)
myLog('epoch\tlr\tloss\tval_loss', path=resultsDir+currRun)

epochs = 210

if not hybridModel:
    model.fit(dTrain, mTrain, shuffle=True, epochs=epochs, batch_size=batch_size,
          validation_data=(dValid, mValid), callbacks=callbacks)
    
elif hybridModel:
    mTrain_latent = np.zeros([len(mTrain), latent_dim], dtype='float32')
    mValid_latent = np.zeros([len(mValid), latent_dim], dtype='float32') 
    if mask_image:
        for i,img in enumerate(mTrain_masked):
            img = img[np.newaxis,:]
            print('... Predicting train image ' + str(i))
            mTrain_latent[i] = np.squeeze(encoder.predict(img))
            #predicted = predicted[::-1]
        for i,img in enumerate(mValid_masked):
            img = img[np.newaxis,:]
            print('... Predicting valid image ' + str(i))
            mValid_latent[i] = np.squeeze(encoder.predict(img))
            #predicted = predicted[::-1] 
    else:
        for i,img in enumerate(mTrain):
            img = img[np.newaxis,:]
            print('... Predicting train image ' + str(i))
            mTrain_latent[i] = np.squeeze(encoder.predict(img))
            #predicted = predicted[::-1]
        for i,img in enumerate(mValid):
            img = img[np.newaxis,:]
            print('... Predicting valid image ' + str(i))
            mValid_latent[i] = np.squeeze(encoder.predict(img))
            #predicted = predicted[::-1]
        
    model.fit(dTrain, [mTrain, mTrain_latent], shuffle=True, epochs=epochs, batch_size=batch_size,
              validation_data=(dValid, [mValid, mValid_latent]), callbacks=callbacks)   

end = datetime.datetime.now()
elapsed = end-start
myPrint('...End: {}'.format(end.ctime()[:-5]), path=resultsDir+currRun)
myPrint('...Train time: {}'.format(elapsed), path=resultsDir+currRun)
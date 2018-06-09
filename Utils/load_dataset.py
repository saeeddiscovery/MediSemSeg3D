# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 16:04:07 2016

@author: Saeed Mhq
"""
# In the name of GOD

import os, glob
from random import shuffle
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
if __name__ == '__main__':
    from utils import myPrint
else:
    from .utils import myPrint

def split_list(img_list, msk_list, split, shuffleList=True):
    if shuffleList:
        s = list(zip(img_list, msk_list))
        shuffle(s)
        img_list, msk_list = zip(*s)
    n_train = int(len(img_list) * split)
    train_list_img = img_list[:n_train]
    valid_list_img = img_list[n_train:]
    train_list_msk = msk_list[:n_train]
    valid_list_msk = msk_list[n_train:]
    return train_list_img, valid_list_img, train_list_msk, valid_list_msk

def load_images(images_list, padSize, ScaleFactor):
    noImages = int(images_list.__len__())
    
    if (noImages == 0):
        return []
    
    dImages = list()
    dImages_zoomed = list()

    for i,file in enumerate(images_list):
        imgFilename = os.path.join(file)
        imgInput = sitk.ReadImage(imgFilename)
        img = sitk.GetArrayFromImage(imgInput)
        img = np.asarray(img, dtype='float32')
#        img = np.rot90(img)
#        img = np.fliplr(img)
        img = np.flipud(img)
        if padSize > 0:
            dImages.append(np.pad(img, padSize, 'edge'))
        else:
            dImages.append(img)
        
#    dTrain_pad = np.zeros(shape=[noTrain,img.shape[0]+padSize*2,img.shape[1]+padSize*2,img.shape[2]+padSize*2], dtype='float32')
#    dTrain_zoomed = np.zeros(shape=[noTrain,int(dTrain_pad.shape[1]*ScaleFactor),int(dTrain_pad.shape[2]*ScaleFactor),int(dTrain_pad.shape[3]*ScaleFactor)], dtype='float32')
#    for i in range(len(dTrain)):
#        dTrain_pad[i,:,:,:] = np.pad(dTrain[i], padSize, 'constant')

#    dImages = rescale(dImages)

    if ScaleFactor == None:
        dImages = np.asarray(dImages)
        dImages = np.array(dImages[:,:,:,:,np.newaxis], dtype='float32')
    else:
        for i in range(noImages):
            dImages_zoomed.append(ndimage.zoom(dImages[i], ScaleFactor, order=0))
#            dTrain_zoomed[i,:,:,:] = ndimage.zoom(dTrain[i], ScaleFactor, order=0) #Pooling
        dImages = np.asarray(dImages_zoomed)
        dImages = dImages[:,:,:,:,np.newaxis]
        
    return dImages

def save_list(images_list, fileName):
    # save train/validation files in a text file
    if not os.path.exists(os.path.dirname(fileName)):
        os.mkdir(os.path.dirname(fileName))
    i=0
    with open(fileName, 'w') as imagesListFile:
        for item in images_list:
            imagesListFile.write(str(i)+"\t{}\n".format(item))
            i+=1

def prepare_dataset(datasetDir, split=0.8, padSize=0, shuffle=True, scaleFactor=None, logPath='.'):
    """ 
    Function that loads 3D medical image data
    and prepare it for training
    
    Arguments:
        - datasetDir: The directory that contains all dataset images
        - split: The ratio (0-1) of images for the training data
        - padSize: The number of voxels to pad
            `Default`: zero (no padding)
        - shuffle_list: Defines if the list of files should be shuffled or not
                `Default`: True (shuffle the images list)
        - ScaleFactor: Defines the scale of the data (0.5 -> 1/2 size)
            `Default`: None (No scale)
    """
    for dataset in ['\\Dataset_3DirCab_Preprocessed']:
        img_addrs = glob.glob(os.path.join(datasetDir+dataset, 'liver_*-patient.nii.gz'))
        msk_addrs = glob.glob(os.path.join(datasetDir+dataset, 'liver_*-liver.nii.gz'))

# Second method for Trian/Valid images list creation
    train_list_img, valid_list_img, train_list_msk, valid_list_msk = split_list(img_addrs, msk_addrs, split=split, shuffleList=shuffle)

    dTrain = load_images(train_list_img, padSize, scaleFactor)
    mTrain = load_images(train_list_msk, padSize, scaleFactor)
    save_list(train_list_img, logPath+'/reports/train_list.txt')
    myPrint('------------<  Dataset Info >------------', path=logPath)
    myPrint('...Train images:      {0}'.format(len(dTrain)), path=logPath)
    dValid = load_images(valid_list_img, padSize, scaleFactor)
    mValid = load_images(valid_list_msk, padSize, scaleFactor)
    save_list(valid_list_img, logPath+'/reports/valid_list.txt')
    myPrint('...Validation images: {0}'.format(len(dValid)), path=logPath)

    return dTrain, mTrain, dValid, mValid


def normalize(data):
    count = len(data)
    for i in range(count) :
        data[i] = np.add(data[i], -np.mean(data[i][:]))
        data[i] /= np.std(data[i][:])
#        data[i] = np.add(data[i], -np.min(data[i]))
#        data[i] /= np.add(np.max(data[i]), -np.min(data[i]))
        #data[i,:] = np.add(data[i,:], -np.min(data[i,:]))
    return data

def rescale(data):
    for i, data_rescaled in enumerate(data):
        data[i] /= np.max(data_rescaled[:])
#        data[i] -= np.mean(data_rescaled[:])
#        data[i] /= np.std(data_rescaled[:])
        data[i] -= 0.5
        data[i] *= 2.
        data[i] = np.asarray(data[i], 'float32')
    return data

if __name__ == '__main__':
    datasetDir = r'C:\Users\user\Desktop\Saeed\MedSemSeg3D\Dataset'
    padSize = 0
    dTrain, mTrain, dValid, mValid = prepare_dataset(datasetDir)
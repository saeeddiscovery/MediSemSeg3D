# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:09:50 2018

@author: Saeed Mhq
"""

#In the name of GOD

import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1.):
    K = tf.keras.backend
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def ConvBlock(x, filters, kernel, dilation_rate=(1,1,1)):
    x = tf.keras.layers.Conv3D(filters, kernel, dilation_rate=dilation_rate,
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def UpConvBlock(x, filters, kernel, strides):
    x = tf.keras.layers.Conv3DTranspose(filters, kernel, strides,
                                        padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def DilatedNet_3D(img_size, downsize_filters_factor=2):
    inLayer = tf.keras.layers.Input(shape=img_size)
    conv1 = ConvBlock(inLayer, filters=int(16/downsize_filters_factor), kernel=(3,3,3))
    x = ConvBlock(conv1, filters=int(32/downsize_filters_factor), kernel=(3,3,3), dilation_rate=(2,2,2))
    conv2 = ConvBlock(x, filters=int(32/downsize_filters_factor), kernel=(3,3,3))
    x = ConvBlock(conv2, filters=int(64/downsize_filters_factor), kernel=(3,3,3), dilation_rate=(2,2,2))
    conv3 = ConvBlock(x, filters=int(64/downsize_filters_factor), kernel=(3,3,3))
    x = ConvBlock(conv3, filters=int(128/downsize_filters_factor), kernel=(3,3,3), dilation_rate=(2,2,2))
    last = ConvBlock(x, filters=int(256/downsize_filters_factor), kernel=(3,3,3))
    
    upconv1 = ConvBlock(x, filters=int(128/downsize_filters_factor), kernel=(3,3,3))
    x = tf.keras.layers.concatenate([upconv1, conv3], axis=-1)
    upconv2 = ConvBlock(x, filters=int(64/downsize_filters_factor), kernel=(3,3,3))
    x = tf.keras.layers.concatenate([upconv2, conv2], axis=-1)
    upconv3 = ConvBlock(x, filters=int(32/downsize_filters_factor), kernel=(3,3,3))
    x = tf.keras.layers.concatenate([upconv3, conv1], axis=-1)
    x = ConvBlock(x, filters=int(16/downsize_filters_factor), kernel=(3,3,3))
    x = tf.keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), padding='same', activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inLayer, outputs=x)
    
    return model

if __name__ == '__main__':
    img_size = (128,128,128,1)
    model = UNet_3D(img_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss = dice_coef_loss, metrics = ['accuracy'])
    model.summary()    
    tf.keras.utils.plot_model(model, to_file='./UNet_3D_Model.png', show_shapes=True)
    print('Model structure saved to UNet_3D_Model.png')

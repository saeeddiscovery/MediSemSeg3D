# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:09:50 2018

@author: Saeed Mhq
"""

#In the name of GOD

import tensorflow as tf
Lambda = tf.keras.layers.Lambda

def ConvBN(xin, filters, dilation, name):
    kernel=(3,3,3)
    x = tf.keras.layers.Conv3D(filters, kernel, dilation_rate=dilation,
                               padding='same', activation=None, name=name)(xin)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

def BNConv(xin, filters, dilation, name):
    kernel=(3,3,3)
    x = tf.keras.layers.BatchNormalization()(xin)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.Conv3D(filters, kernel, dilation_rate=dilation,
                               padding='same', activation=None, name=name)(x)
    return x

def DilatedNet_3D(img_size, d=1):
    inLayer = tf.keras.layers.Input(shape=img_size, name='input')
    conv1 = ConvBN(inLayer, filters=16/d, dilation=1, name='Conv1')
    x = BNConv(conv1, filters=32/d, dilation=1, name='Conv2')
    conv3 = BNConv(x, filters=32/d, dilation=2, name='Conv3')
    x = BNConv(conv3, filters=32/d, dilation=4, name='Conv4')
    conv5 = BNConv(x, filters=32/d, dilation=8, name='Conv5')
    x = BNConv(conv5, filters=32/d, dilation=16, name='Conv6')

#    conv4 = ConvBN(x, filters=256/d, dilation=1, name='Conv4')
    
    conv7 = BNConv(x, filters=32/d, dilation=1, name='Conv7')
    x = tf.keras.layers.concatenate([conv7, conv5], axis=-1, name='Concat1')
    conv8 = BNConv(x, filters=32/d, dilation=1, name='Conv8')
    x = tf.keras.layers.concatenate([conv8, conv3], axis=-1, name='Concat2')
    conv9 = BNConv(x, filters=32/d, dilation=1, name='Conv9')
    x = tf.keras.layers.concatenate([conv9, conv1], axis=-1, name='Concat3')
    conv10 = BNConv(x, filters=16/d, dilation=1, name='Conv10')   
    x = tf.keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1),
                               activation='sigmoid', name='Last')(conv10)
    
    model = tf.keras.Model(inputs=inLayer, outputs=x)
    
    return model

def DilatedNet_3D_2(img_size, d=1):
    inLayer = tf.keras.layers.Input(shape=img_size, name='input')
    n1 = ConvBN(inLayer, filters=16/d, dilation=1, name='n1')
    
    x = BNConv(n1, filters=16/d, dilation=1, name='n2')
    n3 = BNConv(x, filters=16/d, dilation=1, name='n3')
    x = tf.keras.layers.concatenate([n1, n3], axis=-1, name='c1')
    
    x = BNConv(x, filters=16/d, dilation=1, name='n4')
    n5 = BNConv(x, filters=16/d, dilation=1, name='n5')
    x = tf.keras.layers.concatenate([n3, n5], axis=-1, name='c2')
    
    x = BNConv(x, filters=32/d, dilation=2, name='n8') 
    n9 = BNConv(x, filters=32/d, dilation=2, name='n9')
    x = tf.keras.layers.concatenate([n5, n9], axis=-1, name='c4')
    
    x = BNConv(x, filters=32/d, dilation=2, name='n10') 
    n11 = BNConv(x, filters=32/d, dilation=2, name='n11')
    x = tf.keras.layers.concatenate([n9, n11], axis=-1, name='c5')
    
    x = BNConv(x, filters=32/d, dilation=4, name='n14') 
    n15 = BNConv(x, filters=32/d, dilation=4, name='n15')
    x = tf.keras.layers.concatenate([n11, n15], axis=-1, name='c7')
    
    x = BNConv(x, filters=32/d, dilation=4, name='n16') 
    n17 = BNConv(x, filters=32/d, dilation=4, name='n17')
    x = tf.keras.layers.concatenate([n15, n17], axis=-1, name='c8')
    
    x = tf.keras.layers.Conv3D(filters=2/d, kernel_size=(1, 1, 1),
                               activation='sigmoid', name='Last')(x)
    
    model = tf.keras.Model(inputs=inLayer, outputs=x)
    
    return model

if __name__ == '__main__':
    img_size = (128,128,128,1)
    model = DilatedNet_3D_2(img_size)
    model.summary()    
    tf.keras.utils.plot_model(model, to_file='./DilatedNet_3D_Model.png', show_shapes=True)
    print('Model structure saved to DilatedNet_3D_Model.png')

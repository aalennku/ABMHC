#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07 13:30:00 2020
@author: Alan J.X. Guo
"""

import argparse
import scipy.io as sio
import numpy as np
import random
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
sys.path.append('./VCA')
from VCA import vca

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Softmax, Conv1D, Flatten, Add, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping


class En_De(keras.layers.Layer):
    def __init__(self, endmembers_init, **kwargs):
        self.emb_init = np.copy(endmembers_init)
        self.channels = self.emb_init.shape[-1]
        super(En_De, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.emb_wt = self.add_weight(name='emb_wt', 
                                      shape=self.emb_init.shape,
                                      initializer=tf.constant_initializer(self.emb_init),
                                      trainable=True)
        super(En_De, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        return [K.dot(inputs,self.emb_wt),tf.einsum('ij,jk->ijk',inputs,self.emb_wt)]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.channels),(input_shape[0],input_shape[1],self.channels)]

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',
                    help='Path of HSI datamat')
parser.add_argument('-k','--key',
                    default=None,
                    help='key of the HSI tensor in the matlab datamat, valid when using *.mat file')
args = parser.parse_args()

if os.path.splitext(args.path)[-1] == '.npy':
    print('load {0}.'.format(args.path))
    data_mat = np.load(args.path)
elif os.path.splitext(args.path)[-1] == '.mat':
    print('load {0} from {1}.'.format(args.key,args.path))
    data_mat = sio.loadmat(args.path)
    assert args.key in data_mat
    data_mat = data_mat[args.key]

def abs_softmax(x):
    return tf.math.abs(x)/tf.math.reduce_sum(tf.math.abs(x),
                                             axis=-1,
                                             keepdims=True)

R = 16
CHANNELS = data_mat.shape[-1]
LAMBDA = 0.5
EPOCHS = 200
BATCH_SIZE = 256

vca_x = (data_mat.reshape(-1,CHANNELS).T-np.min(data_mat))/np.max(data_mat)
endmembers, no, reconstruct = vca(vca_x,R)

inputs = Input(shape=(CHANNELS,1))
e1 = Conv1D(512,3,data_format='channels_last',use_bias=True,activation='relu')(inputs)
e2 = Conv1D(128,3,data_format='channels_last',use_bias=True,activation='relu')(e1)
e2 = Flatten()(e2)
e3 = Dense(R,activation=abs_softmax)(e2)
ende = En_De(endmembers.T)
de, de_spand = ende(e3)
d1 = Conv1D(256,1,data_format='channels_first',use_bias=True, activation='relu')(de_spand)
d2 = Conv1D(256,1,data_format='channels_first',use_bias=True, activation='relu')(d1)
d3 = Conv1D(16,1,data_format='channels_first',use_bias=True, activation='relu')(d2)
d4 = Conv1D(16,1,data_format='channels_first',use_bias=True, activation='relu')(d3)
d5 = Conv1D(1,1,data_format='channels_first',use_bias=True, activation='linear')(d4)
d5 = Flatten()(d5)
output = Add()([d5*(1-LAMBDA),de*LAMBDA])
autoencoder = keras.models.Model(inputs=inputs, outputs=output)

ae_x = np.copy(vca_x.T)
np.random.shuffle(ae_x)
ae_x = ae_x[:,:,np.newaxis]

optimizer = keras.optimizers.Adam(lr=0.001)
ende.trainable = True
autoencoder.compile(optimizer, loss='mean_squared_error')

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=30,
                               monitor='loss',
                               min_delta=1e-8,
                               min_lr=1e-6,verbose=True)
earlystopping = EarlyStopping(monitor='loss', min_delta=1e-8, patience=50, 
                              verbose=1, mode='auto', baseline=None, 
                              restore_best_weights=True)
callbacks = [lr_reducer, earlystopping]
history = autoencoder.fit(x=[ae_x],y=[ae_x],batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=callbacks, 
                  shuffle=True)

re = autoencoder.predict(ae_x,batch_size=1024)
diff = re - ae_x.reshape(re.shape)
print("reconstruction error: {0}".format(np.mean(np.mean(np.square(diff),axis=1))))

encoder = keras.models.Model(inputs=inputs, outputs=e3)
abundance = encoder.predict(x=[ae_x],batch_size=1024)
shape = list(data_mat.shape)
shape[-1] = R
abundance = abundance.reshape(shape)
save_path = os.path.splitext(args.path)[0] + '_abundance.npy'
np.save(save_path,abundance)
print('abundance saved to {0}.'.format(save_path))

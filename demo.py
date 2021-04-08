#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07 13:30:00 2020
@author: Alan J.X. Guo
"""

from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import scipy.io as sio
import numpy as np
import random
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Input, Dense, Softmax, Conv1D, Flatten, Add, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import json
import os

def chop(img, gt, half_size=5):
    _img = np.concatenate([img[0:1]]*half_size+[img]+[img[-1:]]*half_size,axis=0)
    _img = np.concatenate([_img[:,0:1]]*half_size+[_img]+[_img[:,-1:]]*half_size, axis=1)
    _gt = np.concatenate([gt[0:1]]*half_size+[gt]+[gt[-1:]]*half_size,axis=0)
    _gt = np.concatenate([_gt[:,0:1]]*half_size+[_gt]+[_gt[:,-1:]]*half_size, axis=1)
    print(_img.shape,_gt.shape)
    data_dict = dict()
    seq_data = []
    for idx_i in tqdm(range(_img.shape[0])):
        for idx_j in range(_img.shape[1]):
            label = int(_gt[idx_i,idx_j])
            patch = _img[max(idx_i-half_size,0):min(idx_i+half_size+1,_img.shape[0]),
                        max(idx_j-half_size,0):min(idx_j+half_size+1,_img.shape[1])]
            if patch.shape[:2] == (half_size*2+1,half_size*2+1):
                seq_data.append(patch)
                if not label in data_dict:
                    data_dict[label] = []
                data_dict[label].append(patch)
    return data_dict, np.array(seq_data)

def evaluate(test_data_dict,num_classes,a,b):
    conf_matrix = []
    save_list = []
    for key in tqdm(sorted(test_data_dict)):
        ans = m_classification.predict(np.array(test_data_dict[key]),
                                       batch_size=2048)
        ans_vector = [0]*num_classes
        for item in ans:
            item_ = np.argmax(item[a:b])
            ans_vector[item_] += 1
        conf_matrix.append(ans_vector)
    accuracy = []
    corrected = 0
    total = 0
    for idx in range(len(conf_matrix)):
        accuracy.append(conf_matrix[idx][idx]/sum(conf_matrix[idx]))
        corrected += conf_matrix[idx][idx]
        total += sum(conf_matrix[idx])
    oa = corrected/total
    conf_matrix = np.array(conf_matrix)
    pe = (np.sum(np.sum(conf_matrix,axis=0)*np.sum(conf_matrix,axis=1)))/np.sum(conf_matrix)/np.sum(conf_matrix)
    aa = np.mean(accuracy)
    kappa = (oa-pe)/(1-pe)

    save_list.append(accuracy+[oa,aa,kappa])
    print(oa,aa,kappa)
    return save_list


HALFSIZE = 5
PICK_NUMBER_LIST = [200,200,200,0.2]
SEED = 0

pc_data_mat = np.load('DATASETS/pc/Pavia_abundance.npy')
pc_data_mat = pc_data_mat - np.mean(pc_data_mat)
pc_data_mat_gt = sio.loadmat('DATASETS/pc/Pavia_gt.mat')['pavia_gt']

pu_data_mat = np.load('DATASETS/pu/PaviaU_abundance.npy')
pu_data_mat = pu_data_mat - np.mean(pu_data_mat)
pu_data_mat_gt = sio.loadmat('DATASETS/pu/PaviaU_gt.mat')['paviaU_gt']

sa_data_mat = np.load('DATASETS/sa/Salinas_corrected_abundance.npy')
sa_data_mat = sa_data_mat - np.mean(sa_data_mat)
sa_data_mat_gt = sio.loadmat('DATASETS/sa/Salinas_gt.mat')['salinas_gt']

ho_data_mat = np.load('DATASETS/houston2018/Houston2018_crop_abundance.npy')
ho_data_mat = ho_data_mat - np.mean(ho_data_mat)
ho_data_mat_gt = np.load('DATASETS/houston2018/Houston2018_gt_half.npy')


pc_data_dict, pc_seq_data = chop(pc_data_mat,pc_data_mat_gt,half_size=HALFSIZE)
pu_data_dict, pu_seq_data = chop(pu_data_mat,pu_data_mat_gt,half_size=HALFSIZE)
sa_data_dict, sa_seq_data = chop(sa_data_mat,sa_data_mat_gt,half_size=HALFSIZE)
ho_data_dict, ho_seq_data = chop(ho_data_mat,ho_data_mat_gt,half_size=HALFSIZE)

np.random.seed(SEED)
data_dict_list= [pc_data_dict,pu_data_dict,sa_data_dict,ho_data_dict]
name_list = ['pc','pu','sa','ho']
pc_test_data_dict = {}
pu_test_data_dict = {}
sa_test_data_dict = {}
ho_test_data_dict = {}
test_data_dict_list = [pc_test_data_dict,pu_test_data_dict,sa_test_data_dict,ho_test_data_dict]
key_shift = 0
train_data_dict = {}
for data_dict,PICK_NUMBER,test_data_dict in zip(data_dict_list,PICK_NUMBER_LIST,test_data_dict_list):
    PICK_NUMBER_ = PICK_NUMBER
    for key in tqdm(data_dict.keys()):
        PICK_NUMBER = PICK_NUMBER_
        if key == 0:
            continue

        real_key = key+key_shift-1
        if not real_key in train_data_dict:
            train_data_dict[real_key] = []

        if not key in test_data_dict:
            test_data_dict[key] = []

        np.random.shuffle(data_dict[key])
        if type(PICK_NUMBER) is float:
            PICK_NUMBER = int(len(data_dict[key])*PICK_NUMBER)
        train_data_dict[real_key] += data_dict[key][:PICK_NUMBER]
        test_data_dict[key] += data_dict[key][PICK_NUMBER:]
        if PICK_NUMBER <= 300:
            train_data_dict[real_key] += [item.transpose((1,0,2)) for item in data_dict[key][:PICK_NUMBER]] 
            train_data_dict[real_key] += [np.flip(item,axis=0) for item in data_dict[key][:PICK_NUMBER]] 
            train_data_dict[real_key] += [np.flip(item,axis=1) for item in data_dict[key][:PICK_NUMBER]]

    key_shift += len(data_dict.keys())-1

############# data
train_data = []
train_label = []
NUM_IN_CLASS = 3200
NB_CLASSES = len(train_data_dict.keys())

for key in tqdm(sorted(train_data_dict.keys())):
    item_a_list = np.copy(train_data_dict[key])

    if len(item_a_list)+1 < NUM_IN_CLASS:
        item_a_list = np.concatenate([item_a_list]*(NUM_IN_CLASS//len(item_a_list)+1),
                                axis=0)
    item_b_list = np.copy(item_a_list)

    np.random.shuffle(item_a_list)
    np.random.shuffle(item_b_list)
    count = len(train_data_dict[key])
    train_data += item_a_list.tolist()[:NUM_IN_CLASS]
    train_label += [key] * len(item_a_list[:NUM_IN_CLASS])

train_data = np.array(train_data)
train_label = np.array(train_label)
train_label = to_categorical(train_label, num_classes=NB_CLASSES)

# data_end model start

NUMBER_CLASSES = NB_CLASSES
R = 16
PATCH_SIZE = HALFSIZE * 2 + 1
EPOCHS = 1
BATCH_SIZE = 128

input_c = Input(shape=(PATCH_SIZE,PATCH_SIZE,R))
cc1 = Conv2D(64,3,strides=(1,1),activation='relu')(input_c)
cc2 = Conv2D(32,3,strides=(1,1),activation='relu')(cc1)
cc2 = Conv2D(16,3,strides=(1,1),activation='relu')(cc2)
cc2 = Flatten()(cc2)
dd3 = cc2
output_c = Dense(NUMBER_CLASSES,activation='softmax')(dd3)
m_classification = keras.models.Model(inputs=input_c, 
                                        outputs=output_c)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                monitor='val_loss',
                                patience=10,
                                min_delta=0.001,
                                min_lr=0.5e-5,verbose=True)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, 
                                patience=50, verbose=1, mode='auto', 
                                baseline=None, 
                                restore_best_weights=True)
callbacks = [lr_reducer, earlystopping]

optimizer = keras.optimizers.Adam(lr=0.001)
m_classification.compile(optimizer, loss='categorical_crossentropy', metrics=['acc'])
from sklearn.utils import shuffle
train_data,train_label = shuffle(train_data,train_label)
m_classification.fit(x=[train_data],
                        y=[train_label],
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS, verbose=1,
                        validation_split=0.05,
                        validation_freq=1,
                        shuffle=True,callbacks=callbacks)

####model end
num_class_list = [9,9,16,20]
a_list = [0,9,18,34]
b_list = [9,18,34,54]

results = dict()
for name,test_data_dict,num_class,a,b in zip(name_list,test_data_dict_list,num_class_list,a_list,b_list):
    ans = evaluate(test_data_dict,num_class,a,b)
    if name not in results:
        results[name] = []
    results[name].append(ans[:])
    print("{0}: {1}".format(name,ans))

with open('results.json','w') as f:
    f.write(json.dumps(results))
print('results saved to {0}.'.format('results.json'))
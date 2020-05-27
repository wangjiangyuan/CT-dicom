# -*- coding:utf-8 -*-

# 在这个文件中我将展示如何训练模型，stage1基本没用，仅用stage2



from utilize_190805 import *
import tensorflow as tf
import os
import numpy as np
import unet
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K



path_model='/public/home/liulizuo/new_step/'
path_DATA='/public/home/liulizuo/TrainData_1226/'



def get_npy_data(_DATA_DIR, _DATA_LIST, shape):
    image = np.zeros([len(_DATA_LIST), shape[0], shape[1], shape[2], 1])
    mask = np.zeros([len(_DATA_LIST), shape[0], shape[1], shape[2], 1])
    for i in range(len(_DATA_LIST)):
        image[i, :, :, :, :] = np.load(_DATA_DIR + '/CT/' + _DATA_LIST[i])
        mask[i, :, :, :, :] = np.load(_DATA_DIR + '/mask/' + _DATA_LIST[i])
    return image, mask

def read_data(_DATA_DIR, _DATA_LIST, batch_size):
    while 1:
        example = np.load(_DATA_DIR + '/mask/' + _DATA_LIST[0])
        for i in range(0, len(_DATA_LIST), batch_size):
            image, mask = get_npy_data(_DATA_DIR, _DATA_LIST[i:i + batch_size], example.shape)
            yield ({'inputs': image}, {'outputs': mask})

def split(_DATA_DIR, train_ratio=0.92, test_ratio=0.02):
    DATA_LIST = os.listdir(_DATA_DIR + 'mask')
    train = DATA_LIST[0:int(len(DATA_LIST) * train_ratio)]
    test = DATA_LIST[int(len(DATA_LIST) * train_ratio):int(len(DATA_LIST) * (train_ratio + test_ratio))]
    vali = DATA_LIST[int(len(DATA_LIST) * (test_ratio + train_ratio)):len(DATA_LIST)]

    return train, test, vali


def train_stage1(img_shape,epochs=1,batch_size=1,organ='Mandible'):

    model = unet.model(img_shape)
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=unet.bce_dice_loss, metrics=[unet.dice_loss])

    model.summary()
    DATA_DIR = path_DATA+organ+'/shrink/'

    train_set, test_set, vali_set = split(DATA_DIR)

    save_model_path = path_model+organ+'_step1.hdf5'
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True,
                                            verbose=1)

    history = model.fit_generator(generator=read_data(DATA_DIR, train_set, batch_size),
                                  steps_per_epoch=int(len(train_set) / batch_size) + 1,
                                  epochs=epochs, validation_data=read_data(DATA_DIR, vali_set, batch_size),
                                  validation_steps=int(len(vali_set) / batch_size) + 1,
                                  callbacks=[cp])
    K.clear_session()
    del model

def train_stage2(img_shape,epochs=1,batch_size=1,organ='Mandible'):

    model = unet.model(img_shape)

    #adam = tf.keras.optimizers.Adam(lr=0.0003)
    adam = tf.keras.optimizers.Adam(lr=0.00012, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True)
    model.compile(optimizer=adam, loss=unet.bce_dice_loss, metrics=[unet.dice_loss])



    model.summary()
    DATA_DIR = path_DATA+organ+'/cutted/'

    train_set, test_set, vali_set = split(DATA_DIR)

    save_model_path = path_model+organ+'_step3.hdf5'
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True,
                                            verbose=1)

    history = model.fit_generator(generator=read_data(DATA_DIR, train_set, batch_size),
                                  steps_per_epoch=int(len(train_set) / batch_size) + 1,
                                  epochs=epochs, validation_data=read_data(DATA_DIR, vali_set, batch_size),
                                  validation_steps=int(len(vali_set) / batch_size) + 1,
                                  callbacks=[cp])
    K.clear_session()

def delete_file(organpath):
    import os
    pathlist = os.listdir(organpath)
    for path in pathlist:
        if '.npy' in path:
            os.remove(organpath+path)




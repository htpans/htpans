#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import skimage.io as skio
from keras.utils import np_utils
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers 
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, LeakyReLU, BatchNormalization, Concatenate
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model, Model
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from time import time
import gc
import keras.backend as K
from keras.layers import AveragePooling2D, merge, Reshape, Lambda, GlobalAveragePooling2D#, Merge
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras import optimizers
import imgaug as ia
from imgaug import augmenters as iaa
import math
from edclasses.ImageDataGenerator_16 import ImageDataGenerator16bit
from keras.engine import Layer, InputSpec
from keras import initializers as initializations
from keras.engine import Layer, InputSpec
from keras import initializers as initializations

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
import argparse
import sys
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

argparser = argparse.ArgumentParser(
    description='Train Various Keras Prebuilt Models')

argparser.add_argument(
    '--build_model',
    action='store_true',
    help='build models from weights')

argparser.add_argument(
    '--train_all',
    action='store_true',
    help='retrain all models')
    
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def VGG_16():
    model_save_path = "/content/gdrive/My Drive/LD_model_train/Custom_VGG_16.h5"
    graph_name = 'Custom_VGG_16'
    image_shape = (1,100,100)
    
    model = Sequential()
    #model.add(ZeroPadding2D((1,1),input_shape=(1,100,100),dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3, activation='relu',input_shape=(1,100,100)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    #accuracy = 77% after removal of bad images and 200 epochs
    #Loss = 0.6
    return model,model_save_path,graph_name, image_shape[1:2], "channels_first"

def create_VGG19():
    model_save_path = "/content/gdrive/My Drive/LD_model_train/LDVGG19.h5"
    image_shape=(1,100,100)
    graph_name = 'LDVGG19'
    model = VGG19(include_top=True, weights=None, input_tensor = Input(shape=image_shape), input_shape = image_shape, classes=2)
    if os.path.isfile(model_save_path):
       model.load_weights(model_save_path)
        
    return model,model_save_path,graph_name, (100,100), "channels_first"


def create_InceptionV2():
    model_save_path = "/content/gdrive/My Drive/LD_model_train/LDInceptionV2.h5"
    image_shape = (1,200,200)
    graph_name = 'LDInceptionV2'
    model = InceptionResNetV2(include_top=True, weights=None, input_tensor = Input(shape=image_shape), input_shape = image_shape, classes=2)
    if os.path.isfile(model_save_path):        
        model.load_weights(model_save_path)  
        
    return model,model_save_path,graph_name, (200,200), "channels_first"

def create_InceptionV3():
    model_save_path = "/content/gdrive/My Drive/LD_model_train/LDInceptionV3.h5"
    model = InceptionV3(include_top=True, weights=None, input_tensor = Input(shape=(1,250, 250)), input_shape = (1,250,250), classes=2)
    if os.path.isfile(model_save_path):
       model.load_weights(model_save_path)
           
    return model,model_save_path, 'LDInceptionV3', (250,250), "channels_first"

def create_VGG16():
    model_save_path = "/content/gdrive/My Drive/LD_model_train/LDVGG16.h5"
    image_shape = (1,100,100)
    graph_name = 'LDVGG16'
    model = VGG16(include_top=True, weights=None, input_tensor = Input(shape=image_shape), input_shape = image_shape, classes=2)
    if os.path.isfile(model_save_path):        
        model.load_weights(model_save_path)  
        
    return model,model_save_path,graph_name, (100,100), "channels_first"

def create_ResNet50():
    model_save_path = "/content/gdrive/My Drive/LD_model_train/LDResNet50.h5"
    image_shape = (1,200,200)
    graph_name = 'LDResNet50'
    model = ResNet50(include_top=True, weights=None, input_tensor = Input(shape=image_shape), input_shape = image_shape, classes=2)
    if os.path.isfile(model_save_path):        
        model.load_weights(model_save_path)  
        
    return model,model_save_path,graph_name, (200,200), "channels_first" #only need (200,200)

def create_Xception():
    K.set_image_dim_ordering('tf')
    model_save_path = "/content/gdrive/My Drive/LD_model_train/Xception.h5"
    image_shape = (100,100,1)
    graph_name = 'Xception'
    model = Xception(include_top=True, weights=None, input_tensor = Input(shape=image_shape), input_shape = image_shape, classes=2)
    if os.path.isfile(model_save_path):        
        model.load_weights(model_save_path)  
        
    return model,model_save_path,graph_name, (100,100), "channels_last" #only need (200,200), has to be channels last

def create_DenseNet121():    
    #The dense net models use a lot of resources, the batch size needs to be lowered to 64
    model_save_path = "/content/gdrive/My Drive/LD_model_train/DenseNet121.h5"
    image_shape = (1,250,250)
    graph_name = 'DenseNet121'
    model = DenseNet121(include_top=True, weights=None, input_tensor = Input(shape=image_shape), input_shape = image_shape, classes=2)
    if os.path.isfile(model_save_path):        
        model.load_weights(model_save_path)  
        
    return model,model_save_path,graph_name, (250,250), "channels_first" #only need (200,200)
    
def create_DenseNet169():
     #The dense net models use a lot of resources, the batch size needs to be lowered to 32
    model_save_path = "/content/gdrive/My Drive/LD_model_train/DenseNet169.h5"
    image_shape = (1,250,250)
    graph_name = 'DenseNet169'
    model = DenseNet169(include_top=True, weights=None, input_tensor = Input(shape=image_shape), input_shape = image_shape, classes=2)
    if os.path.isfile(model_save_path):        
        model.load_weights(model_save_path)  
        
    return model,model_save_path,graph_name, (250,250), "channels_first" #only need (200,200)

def create_DenseNet201():
     #The dense net models use a lot of resources, the batch size needs to be lowered to 32
    model_save_path = "/content/gdrive/My Drive/LD_model_train/DenseNet201.h5"
    image_shape = (1,250,250)
    graph_name = 'DenseNet201'
    model = DenseNet201(include_top=True, weights=None, input_tensor = Input(shape=image_shape), input_shape = image_shape, classes=2)
    if os.path.isfile(model_save_path):        
        model.load_weights(model_save_path)  
        
    return model,model_save_path,graph_name, (250,250), "channels_first" #only need (200,200)


def load_any_model(path_to_model):
    model = load_model(path_to_model)
    print(model.input.shape)
    shape = model.input.shape
    if shape[1] == 1:
        channel_order = "channels_first"
        input_size = (int(shape[2]),int(shape[3]))
    else:
        channel_order = "channels_last"
        input_size = (int(shape[1]),int(shape[2]))
    return model, channel_order, input_size

def train_all():
    model_dir = "/content/LD_model_Built/"    
    log_dir = "/content/gdrive/My Drive/Training_logs/"    
    
    files = os.listdir(model_dir)
    for f in files:
        print(f)
        prefix = f[:-3]
        model_save_path = "/content/gdrive/My Drive/LD_model_train/" + f
        model = load_model(model_dir + f)
        print(model.input.shape)
        shape = model.input.shape
        if shape[1] == 1:
            channel_order = "channels_first"
            input_size = (int(shape[2]),int(shape[3]))
        else:
            channel_order = "channels_last"
            input_size = (int(shape[1]),int(shape[2]))
        train_datagen = ImageDataGenerator16bit(rescale=65535,
        data_format = channel_order,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=20,
        height_shift_range=20)
        test_datagen = ImageDataGenerator16bit(rescale=65535, data_format = channel_order)

        train_generator = train_datagen.flow_from_directory(
          'Data/train',
          target_size=input_size,
          batch_size=16,
          color_mode = 'grayscale',
          class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
          'Data/validation',
          target_size=input_size,
          batch_size=32,
          color_mode = 'grayscale',
          class_mode='binary')
          
        #parallel_model = ModelMGPU(model , 2)
        learning_rate = 0.00001
        adam = optimizers.Adam(lr=learning_rate, decay=0.01)      
        model.compile(loss='sparse_categorical_crossentropy',optimizer = adam , metrics=['accuracy'])     
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(filepath = model_save_path, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False)
        tb_counter  = len([log for log in os.listdir(os.path.expanduser(log_dir)) if prefix in log]) + 1
        tensorboard = TensorBoard(log_dir=os.path.expanduser(log_dir) + prefix + '_' + str(tb_counter), 
                                  histogram_freq=0, 
                                  #write_batch_performance=True,
                                  write_graph=True, 
                                  write_images=False)     
        model.fit_generator(
             train_generator,
             steps_per_epoch=len(train_generator),
             epochs=500,
             validation_data=validation_generator,
             callbacks = [tensorboard,checkpoint,early_stop],
             verbose = 1)                
                
        train_datagen = None
        test_datagen = None
        train_generator = None
        validation_generator = None
                                   
        #model.save("/content/gdrive/My Drive/LD_model_train/" + f)                
        
        gc.collect()

def main(unused_argv):

      #K.set_image_dim_ordering('th') # this is a channel first dim ordering may have to change for different models //th = channels first tf = channels last
      
      train_all()
            
      
if __name__ == "__main__":
  FLAGS, unparsed = argparser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import skimage.io as skio
import os
from keras.models import load_model
from tqdm import tqdm
import os
from skimage.transform import resize
from edclasses.ImageDataGenerator_16 import ImageDataGenerator16bit
import argparse
import time as timer
from keras.utils import multi_gpu_model
from keras import optimizers
import gc
import csv


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

def write_results(models,acc,loss,time):
    with open("Eval_Data.csv", 'w', newline ='') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=',')
        write_row(data_writer,"Measure",models)
        write_row(data_writer,"Accuracy",acc)
        write_row(data_writer,"Loss",loss)
        write_row(data_writer,"Time",time)

def write_results2(models,acc,loss,time,data):
    with open("Eval_Data.csv", 'w', newline ='') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=',')
        for d,m,a,l,t in zip(data,models,acc,loss,time):     
            data_writer.writerow([d,m,a,l,t])
        

def write_row(data_writer,row_name, data):
    row = row_name
    data_writer.writerow(data)
        

def train_all(model_dir, eval_dir):
    files = os.listdir(model_dir)
    acc = []
    loss = []
    time = []    
    for f in files:
        print(f)
        prefix = f[:-3]        
        model, channel_order, input_size = load_any_model(model_dir + f)
                        
        test_datagen = ImageDataGenerator16bit(rescale=65535, data_format = channel_order)

        validation_generator = test_datagen.flow_from_directory(
          eval_dir,
          target_size=input_size,
          batch_size=32,
          color_mode = 'grayscale',
          class_mode='binary')
          
        parallel_model = multi_gpu_model(model , 2)
        learning_rate = 0.0
        adam = optimizers.Adam(lr=learning_rate, decay=0.01)      
        parallel_model.compile(loss='sparse_categorical_crossentropy',optimizer = adam , metrics=['accuracy'])     
        start = timer.time()        
        score = parallel_model.evaluate_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
        end = timer.time()
        print(score)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        acc = acc + [score[1]]
        loss = loss + [score[0]]
        time = time + [end - start]
        model = None
        parrallel_model = None        
        test_datagen = None        
        validation_generator = None
        gc.collect()
    write_results(files,acc,loss,time)

def train_all2(model_dir, eval_dir):
    files = os.listdir(model_dir)
    dirs = os.listdir(eval_dir)
    acc = ['Accuracy']
    loss = ['Loss']
    time = ['Time']    
    folder = ['Data']
    model_type = ['Model']
    for f in files:        
        print(f)
        prefix = f[:-3]        
        model, channel_order, input_size = load_any_model(model_dir + f)
        parallel_model = multi_gpu_model(model , 2)
        for d in dirs:                
            test_datagen = ImageDataGenerator16bit(rescale=65535, data_format = channel_order)
            validation_generator = test_datagen.flow_from_directory(
                eval_dir + d +'/validation/',
                target_size=input_size,
                batch_size=32,
                color_mode = 'grayscale',
                class_mode='binary')
            learning_rate = 0.0
            adam = optimizers.Adam(lr=learning_rate, decay=0.01)      
            parallel_model.compile(loss='sparse_categorical_crossentropy',optimizer = adam , metrics=['accuracy'])     
            start = timer.time()        
            score = parallel_model.evaluate_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
            end = timer.time()
            print(score)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            acc = acc + [score[1]]
            loss = loss + [score[0]]
            time = time + [end - start]
            folder = folder + [d]
            model_type = model_type + [f]
         
        model = None
        parrallel_model = None        
        test_datagen = None        
        validation_generator = None
        gc.collect()
    write_results2(model_type,acc,loss,time,folder)


def eval_all3(model_dir, eval_dir):
    files = os.listdir(model_dir)
    dirs = os.listdir(eval_dir)
    csvfile = open("Eval_Data_All.csv", 'w', newline ='')
    data_writer = csv.writer(csvfile, delimiter=',')
    data_writer.writerow(["Filename","Status","Model","Folder"]) 
    for f in files:        
        print(f)
        prefix = f[:-3]        
        model, channel_order, input_size = load_any_model(model_dir + f)
        #parallel_model = multi_gpu_model(model , 2)
        for d in dirs:
            print("evaluating: " + "eval_dir" + '/' + d)                
            test_datagen = ImageDataGenerator16bit(rescale=65535, data_format = channel_order)
            validation_generator = test_datagen.flow_from_directory(
                eval_dir + '/' + d,
                target_size=input_size,
                batch_size=32,
                color_mode = 'grayscale',
                class_mode='binary',
                shuffle=False)
            #learning_rate = 0.0
            #adam = optimizers.Adam(lr=learning_rate, decay=0.01)      
            #parallel_model.compile(loss='sparse_categorical_crossentropy',optimizer = adam , metrics=['accuracy'])     
            #model.compile(optimizer = 'Adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            #predictions = parallel_model.predict_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
            predictions = model.predict_generator(validation_generator, steps=None, max_queue_size=1, workers=1, use_multiprocessing=False, verbose=1)
            status = predictions[:,0] >= 0.5            
            filenames = validation_generator.filenames
            for fn,p in zip(filenames,status):
                #st = "False"
                #if p[0] >= 0.5:
                #    st = "True"
                st = p
                data_writer.writerow([fn,st,prefix,d])
            #write_results3()
         
        model = None
        #parrallel_model = None        
        test_datagen = None        
        validation_generator = None
        gc.collect()
    csvfile.close()

argparser = argparse.ArgumentParser(
    description='validate model on a dataset')


argparser.add_argument(
    '-i',
    '--input',
    help='path to a validation folder')

argparser.add_argument(
    '-m',
    '--model',
    help='path to model directory or file')

argparser.add_argument(
    '--eval_all',
    action='store_true',
    help='eval all models')

args = argparser.parse_args()

if args.eval_all:
    eval_all3(args.model,args.input)
else:
    model_save_path = args.model #"LDModel_New/DenseNet201_full.h5"
    model, channel_order, input_size = load_any_model(model_save_path)
    #model.compile(optimizer = 'Adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
    test_datagen = ImageDataGenerator16bit(rescale=65535, data_format = channel_order)

    validation_generator = test_datagen.flow_from_directory(
        args.input, #'/run/user/1000/gvfs/smb-share:server=146.189.163.212,share=share/Imaging_Paper_work/Eric/4.10.18_TubaPlateC/validation',#'data/validation',
        target_size=input_size,
        batch_size=32,
        color_mode = 'grayscale',
        class_mode='binary')


    score = model.evaluate_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    






'''
evalImages = skio.imread_collection(eval_path + '/*.tif',conserve_memory=True)
eval_Images = np.asarray(evalImages)
n = eval_Images.shape[0]
eval_Images = np.swapaxes(eval_Images,2,0)
X_eval = resize(eval_Images,[200,200,n],preserve_range = True)
X_eval = np.swapaxes(X_eval,2,0)
X_eval = X_eval.reshape(X_eval.shape[0],1,200,200)
#X_eval = X_eval.astype('float32')
dt = X_eval.dtype
if dt == np.dtype('uint8'):
    X_eval = X_eval / 255
else:
    X_eval = X_eval / 65535
    
model = load_model(model_save_path)
print("Evaluating " + str(X_eval.shape[0]) + " images")

pred = model.predict(X_eval)
status = pred[:,1] > 0.5

for k in range(0,len(pred)):
	text = evalImages.files[k].split('\\')
	if pred[k,1] > 0.5:
		os.rename(evalImages.files[k],text[0] + "/Dead/" + text[1])
	if pred[k,0] > 0.5:
		os.rename(evalImages.files[k],text[0] + "/Alive/" + text[1])

'''    
'''    
eval = np.copy(X_eval[0:4])
for k in tqdm(range(0,len(X_eval))):
    text = evalImages.files[k].split('\\')
    eval[0] = X_eval[k]
    eval[1] = np.rot90(eval[0],axes = (1,2),k=1)
    eval[2] = np.rot90(eval[0],axes = (1,2),k=2)
    eval[3] = np.rot90(eval[0],axes = (1,2),k=2)    
    pred = model.predict(eval)
    if np.mean(pred[:,0]) >= 0.5:
        os.rename(evalImages.files[k],text[0] + "/Dead/" + text[1][:-4] + ".tif")
    if np.mean(pred[:,1]) >= 0.5:
        os.rename(evalImages.files[k],text[0] + "/Alive/" + text[1][:-4] + ".tif")
'''

print("Finsihed")

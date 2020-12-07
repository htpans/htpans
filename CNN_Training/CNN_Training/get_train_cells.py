#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import *
from utils import BoundBox
from frontend import YOLO
import json
from skimage import io
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import skimage.io as skio
from keras.models import load_model
from skimage.transform import resize

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def load_live_dead_model(save_path):
    model = load_model(save_path)
    print(model)
    return model

def resize_images(images,new_shape):
    n = images.shape[0]
    out = []
    #print(np.max(images[0]))
    for k in range(0,n):
        i = resize(images[k],new_shape,mode='reflect',preserve_range = True)
        #print(np.max(i))
        #exit()
        out = out + [i]
    out = np.array(out)
    out = out.reshape(n,1,new_shape[0],new_shape[1])
    return out 

def get_cell_status(images,model):
    #eval_Images = np.asarray(images)
    #images = np.array(images)
    #print(images.shape)
    X_eval = images.reshape(images.shape[0],1,100,100)
    
    shape = model.input.shape
    if shape[1] == 1:
        channel_order = "channels_first"
        input_size = (int(shape[2]),int(shape[3]))
    else:
        channel_order = "channels_last"
        input_size = (int(shape[1]),int(shape[2]))
        X_eval = np.moveaxis(X_eval,1,3)
    
    w=input_size[1]
    h=input_size[0]
    
    if w != 100 and h != 100:
        X_eval = resize_images(images,(h,w))
    
    X_eval = X_eval / 65535
    
    pred = model.predict(X_eval)
    #print(len(pred))
    status = pred[:,0] >= 0.5
    return status

def remove_dead_cells(images,model,boxes):
    if boxes == None:
        return boxes
    status = get_cell_status(images,model)
    out_box = []
    for box,test in zip(boxes,status):
        if test:
            out_box.append(box)           

    return out_box

def remove_dead_cells2(image,model,boxes):
    #check cells over all timepoints
    if boxes == None:
        return boxes
    out_box = []
    for box in boxes:
        images = get_images2(image,box,100)
        status = get_cell_status(images,model)
        if any(status): #if it was alive at some timepoint    
            out_box.append(box)        

    return out_box

def get_dead_cells(images,model,boxes):
    if boxes == None:
        return boxes
    status = get_cell_status(images,model)
    out_box = []
    for box,test in zip(boxes,status):
        if test == False:
            out_box.append(box)           

    return out_box


def make_output_image(cell_images):
     #will make a final image that all the cell_image timecources can be placed into
     h = cell_images.shape[1]
     w = cell_images.shape[2]
     z = cell_images.shape[0]
     out = np.zeros((h*z,w),dtype = np.uint16)
     for k in range(0,z):
          out[k*h:(k+1)*h,0:w] = cell_images[k]
     
     return out



def scanImage(image,yolo,config):    
    height = image.shape[0]
    width = image.shape[1]
    inc = 416
    step = 316
    boxes2 = []    
    for yPos in range(0,height,step):
        for xPos in range(0,width, step):
            minx = 50            
            maxx = 366
            miny = 50
            maxy = 366
            if xPos + inc >= width:
                minx = xPos + 50
                xPos = width - inc
                minx = minx - xPos
                maxx = 416
            if yPos + inc >= height:
                miny = yPos + 50
                yPos = height - inc
                miny = miny - yPos
                maxy = 416
            if xPos == 0:
                xmin = 0
            if yPos == 0:
                ymin = 0
            image2 = image[yPos:yPos + inc,xPos:xPos + inc]
            #draw_boxes(image2, boxes, config['model']['labels'])
            #print("Shape " + str(image2.shape[1]))
            boxes = yolo.predict(image2)
                       
            #boxes = filter_boxes(xPos,yPos,image2,boxes,Q1x,Q1y)
            boxes = filter_boxes2(xPos,yPos,image2,boxes,minx,maxx,miny,maxy)
            #boxes = remove_edge_boxes(xPos,yPos,image2,image,boxes)
            boxes = reshape_boxes(xPos,yPos,image2,image,boxes)
            boxes2 = boxes2 + boxes
                        #image2 = draw_boxes(image2, boxes, config['model']['labels'])
            #image[yPos:yPos + inc,xPos:xPos + inc] = image2
            
    
    
#(tensorflow) C:\Users\UMW\Desktop\YOLO\basic-yolo-keras-master>python predict.py -c config.json -w full_yolo_cells.h5 -i C:/Users/UMW/Desktop/YOLO/basic-yolo-keras-master/images/C2-1.tif
    print("Box len " + str(len(boxes2)))
    boxes2 = sort_boxes(boxes2)
    print("Sorted Box len " + str(len(boxes2)))
    boxes2 = remove_overlap(image,boxes2)
    print("overlap filtered Box len " + str(len(boxes2)))
    return boxes2

def find_cells(file,yolo,qa_path,config,cell_path,image_name,live_dead,data_writer):
    #yolo is the model, image is the first image int he stack, write_path is where the image with boxes will be written,config is the model info,cell_path is where the cells will be dumped for classification
    image = cv2.imread(file)
    unclassified_cells = scanImage(image,yolo,config)
    if len(unclassified_cells) == 0:
        print("no cells")
        return
    image16 = io.imread(file)
    cell_images = get_images(image16[0],unclassified_cells,100)
    #dead_cells = get_dead_cells(cell_images,live_dead,unclassified_cells)
    #dead_images = get_images(image16[0],dead_cells,100)
    
    classified_cells = remove_dead_cells(cell_images,live_dead,unclassified_cells)
    if len(classified_cells) == 0:
        print("no cells")
        return
    out_image = draw_boxes(np.copy(image), unclassified_cells, config['model']['labels'])
    out_image2 = draw_boxes(np.copy(image), classified_cells, config['model']['labels'])
    cv2.imwrite(qa_path + image_name +'_uc.tif',out_image)
    cv2.imwrite(qa_path + image_name +'_cl.tif',out_image2)
    
    cell_number = 1
    time = 1
    all_images= get_images2(image16,classified_cells,100)
    status = get_cell_status(all_images,live_dead)
    write_status(qa_path + image_name + '.csv',status,image16.shape[0])
    montage = get_montages(image16,classified_cells,100)
    large_montage = make_output_image(montage)
    io.imsave(qa_path + image_name + "montage.tif",large_montage)
    write_data(image_name,data_writer,status,image16.shape[0])
    #write_boxes(unclassified_cells, qa_path + image_name + '.csv',image)
    '''
    for cell in all_images:
        io.imsave(cell_path + image_name +'_c_' + str(cell_number) + '_t_' + str(time) +'.tif',cell)
        time = time + 1
        if time > image16.shape[0]:
             cell_number += 1
             time = 1
             '''
    #for cell in dead_images:
    #    io.imsave(cell_path + image_name + str(cell_number) +'.tif',cell)
    #    cell_number += 1        

def process_directory(image_path,output_path,yolo,config,qa_path,live_dead):
    files = os.listdir(image_path)
    #files = files[26:]
    data_writer,csvfile = prepare_data(qa_path + "Data.csv")
    for f in tqdm(files):
        #image = cv2.imread(image_path + f)
        print(f)
        file = os.path.basename(f)
        image_name = os.path.splitext(file)[0]
        find_cells(image_path + f, yolo,qa_path,config,output_path,image_name,live_dead,data_writer)
    csvfile.close()    


def _main_(args):
 
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print(weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]

        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            
            boxes = yolo.predict(image)            
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  
    else:
        #image = cv2.imread(image_path)
        output_path = "/content/gdrive/My Drive/Publication/unclassified/"
        qa_path = "/content/gdrive/My Drive/Publication/qa/"
        image_path = "/content/gdrive/My Drive/Publication/images/"
        #image_path = "V:/Anthony/12.26.17_transfection_started_viewing12.28/Stacks_aligned/"
        #image_path = "V:/Anthony/12.12.17_12.13.17_transfections_384wellFalcon/12.12.17transfectionPlateA/plateAStacksAligned/"
        #image_path = "V:/Anthony/11.14.17 and 11.15 transfections/11.14.17 transfection PlateD Arbez Method/11.14.17plateD aligned/"
        image_path = "/content/StitchedImages/"
        ld_model_path = "/content/LDModel/LDVGG19.h5"
        image_path   = args.input
        #boxes = yolo.predict(image)
        #image = scanImage(image,yolo,config)        
        #image = draw_boxes(image, boxes, config['model']['labels'])
        #find_cells(yolo,image,"fill",config,ip,"test")
        #print(len(boxes), 'boxes are found')
        live_dead = load_live_dead_model(ld_model_path)
        process_directory(image_path,output_path,yolo,config,qa_path,live_dead)

        #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)



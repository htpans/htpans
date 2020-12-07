#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from utils import reshape_boxes
from utils import remove_edge_boxes
from utils import filter_boxes
from utils import filter_boxes2
from utils import sort_boxes
from utils import remove_overlap
from utils import BoundBox
from frontend import YOLO
import json
import xml.etree.ElementTree as ET
import csv

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

argparser.add_argument(
    '-o',
    '--filter',
    help='restrict to specific channel')

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
                minx = 0
            if yPos == 0:
                miny = 0
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
    boxes2 = sort_boxes(boxes2)
    boxes2 = remove_overlap(image,boxes2)            
    image = draw_boxes(image, boxes2, config['model']['labels'])
    
#(tensorflow) C:\Users\UMW\Desktop\YOLO\basic-yolo-keras-master>python predict.py -c config.json -w full_yolo_cells.h5 -i C:/Users/UMW/Desktop/YOLO/basic-yolo-keras-master/images/C2-1.tif
    return image,boxes2

def build_annotation(annotation_name, annotation_folder, image, boxes, names):
        image_height, image_width, channels = image.shape
                
        root = ET.Element("annotation",{"verified":"yes"})
        folder = ET.SubElement(root,"folder")
        folder.text = annotation_folder
        filename = ET.SubElement(root,"path")
        filename.text = annotation_name
        width = ET.SubElement(root,"width")
        width.text = str(image_width)
        height = ET.SubElement(root,"height")
        height.text = str(image_height)
        for box in boxes:
            x1  = int((box.x - box.w/2) * image.shape[1])
            x2  = int((box.x + box.w/2) * image.shape[1])
            y1  = int((box.y - box.h/2) * image.shape[0])
            y2  = int((box.y + box.h/2) * image.shape[0])
            object_e = ET.SubElement(root,"object")
            name = ET.SubElement(object_e,"name")
            name.text = names[box.get_label()]
            print(name.text)
            print(box.classes)
            bndbox = ET.SubElement(object_e,"bndbox")        
            xmin = ET.SubElement(bndbox,"xmin")
            xmin.text= str(x1)
            ymin = ET.SubElement(bndbox,"ymin")
            ymin.text = str(y1)
            xmax = ET.SubElement(bndbox,"xmax")
            xmax.text=str(x2)
            ymax = ET.SubElement(bndbox,"ymax")
            ymax.text=str(y2)
        
        tree = ET.ElementTree(root)
        #tree.write(annotation_folder + annotation_name)
        print(annotation_folder + annotation_name)
        tree.write(annotation_folder + annotation_name)

def process_image_directory(directory,image_path,yolo,channel):
    images = os.listdir(directory + '/' + image_path)
    output = []
    for i in tqdm(images):
        #print(i)
        if (i[-3:] == 'tif' or i[-4:] == 'tiff') and (channel == 'none' or i.find(channel) > -1):
            #print(i)
            image = cv2.imread(directory + '/' + image_path + '/' + i)
            boxes = yolo.predict(image)      
            output.append([i,len(boxes)])
                
    with open('/content/gdrive/My Drive/qa/'+image_path+'.csv', 'w', newline ='\n') as f:
        writer = csv.writer(f)
        writer.writerows(output)

def has_subdir(directory):
    sub_dir = os.listdir(directory)    
    found = False
    print("searching for sub-directories")
    for s in sub_dir:
        found = os.path.isdir(directory + '/'+s)
        #print(found)
        if found:            
            break
    return found

def get_subdir(directory):
    sub_dir = os.listdir(directory)        
    out = []
    for s in sub_dir:
        if os.path.isdir(directory + '/'+ s):
            out = out + [s]
                
    return out   

def _main_(args):
 
    channel = 'none'
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input
    channel = args.filter

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

    if has_subdir(image_path):
        print("processing directory")
        sub_dirs = get_subdir(image_path)
        for directory in sub_dirs:
            process_image_directory(image_path,directory,yolo,channel)
    
    else:
        print("processing images")
        images = os.listdir(image_path)
        output = []
        for i in images:            
            if i[-3:] == 'tif': #and (channel == 'none' or i.find(channel) > -1):
                print(i)
                image = cv2.imread(image_path + '/' + i)
                #boxes = yolo.predict(image)
                image,boxes = scanImage(image,yolo,config)        
                names = config['model']['labels']
                test="/content/gdrive/My Drive/qa/"
                build_annotation(i[:-4]+".xml", test, image, boxes, names)
                image = draw_boxes(image, boxes, config['model']['labels'])
                print(len(boxes), 'boxes are found')
                output.append([i,len(boxes)])
                #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
                cv2.imwrite('/content/gdrive/My Drive/qa/' + i, image)
        with open('/content/gdrive/My Drive/qa/cells_identified.csv', 'w', newline ='\n') as f:
            writer = csv.writer(f)
            writer.writerows(output)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

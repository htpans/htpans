import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2
import csv

class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        self.a     = w*h
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def normalize(image):
    
    image /= 65535

    return image

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2
    
    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2
    
    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    
    intersect = intersect_w * intersect_h
    
    union = box1.w * box1.h + box2.w * box2.h - intersect
    
    return float(intersect) / union

def get_area(box):
    return box.a 
    
def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3  

def remove_edge_boxes(xPos,yPos,image,image2,boxes):
    out = []
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])
        if xmin > 0 and xmax < image.shape[1] and ymin > 0 and ymax < image.shape[0]-10:
            out = out + [box]
                
    return out

def filter_boxes(xPos,yPos,image,boxes,Q1x,Q1y):
    out = []
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])
        if xmin > 0 and ymin > 0 and xmin <= Q1x and ymin <= Q1y:
            out = out + [box]
                
    return out

def filter_boxes2(xPos,yPos,image,boxes,minx,maxx,miny,maxy):
    out = []
    for box in boxes:
        x  = int((box.x) * image.shape[1])    
        y  = int((box.y) * image.shape[0])
        if minx < x < maxx and miny < y < maxy:
            out = out + [box]
                
    return out

def sort_boxes(boxes):
    return sorted(boxes, key = get_area,reverse = True)

def reshape_boxes(xPos,yPos,image,image2,boxes):
    
    for box in boxes:
        box.x  = (xPos + (box.x * image.shape[1])) / image2.shape[1]
        box.w  = (box.w * image.shape[1]) / image2.shape[1]
        box.y  = (yPos + (box.y * image.shape[0])) / image2.shape[0]
        box.h  = (box.h * image.shape[0]) / image2.shape[0]
        
    return boxes
def remove_overlap(image, boxes):
    index = 1
    overlap = np.zeros((image.shape[0]+10,image.shape[1]+10),dtype = 'int32')
    out = []
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])
        
        
        #if at least on corner is not equal to zero there is an overlap
        #if the center of the new box overlaps another then don't add it otherwise add it
        if ymin < 0:
            ymin=0
        if xmin < 0:
            xmin = 0
        if ymax > overlap.shape[0]-1:
            ymax = overlap.shape[0]-1
        if xmax > overlap.shape[1]-1:
            xmax = overlap.shape[1]-1
                    
        if overlap[ymin,xmin] == 0 and overlap[ymin,xmax] == 0 and overlap[ymax,xmin] == 0 and overlap[ymax,xmax] == 0:
            overlap[ymin:ymax,xmin:xmax] = index
            out = out + [box]
        elif overlap[int((ymin+ymax)/2),int((xmin+xmax)/2)] == 0:
            out = out + [box]
        index = index + 1

    return out

def draw_boxes(image, boxes, labels):
    number = 1
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        cv2.putText(image, 
                    str(number), 
                    (xmin, ymin - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,255,0), 2)
        number = number + 1
        
    return image  

def write_boxes(boxes, file_name,image):
    with open(file_name, 'w', newline ='') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=',')
        data_writer.writerow(['Xmin','Xmax','Ymin','Ymax'])
        for box in boxes:
            xmin  = int((box.x - box.w/2) * image.shape[1])
            xmax  = int((box.x + box.w/2) * image.shape[1])
            ymin  = int((box.y - box.h/2) * image.shape[0])
            ymax  = int((box.y + box.h/2) * image.shape[0])
            row = [xmin,xmax,ymin,ymax]
            data_writer.writerow(row)

def write_status(file_name,status,time):
    with open(file_name, 'w', newline ='') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=',')
        data_writer.writerow(['Image','Cell','Time','Status'])
        t = 1
        cell = 1
        for k in range(0,len(status)):            
            row = [file_name,cell,t,status[k]]
            data_writer.writerow(row)
            t = t + 1
            if t > time:
                 t = 1
                 cell = cell + 1

def write_status2(file_name,status1,status2,status3,time):
    with open(file_name, 'w', newline ='') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=',')
        data_writer.writerow(['Image','Cell','Time','Status','Status2','Status3'])
        t = 1
        cell = 1
        for k in range(0,len(status1)):            
            row = [file_name,cell,t,status1[k],status2[k],status3[k]]
            data_writer.writerow(row)
            t = t + 1
            if t > time:
                 t = 1
                 cell = cell + 1

def write_status3(file_name,vgg16,vgg19,resnet50,inceptv2,inceptv3,dense121,dense169,dense201,xception,time):
    with open(file_name, 'w', newline ='') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=',')
        data_writer.writerow(['Image','Cell','Time','Status','LDVGG16','LDVGG19','LDResNet50','VDInceptionV2','LDInceptionV3','LDDenseNet121','LDDenseNet169','LDDenseNet201','LDXception'])
        t = 1
        cell = 1
        for k in range(0,len(vgg16)):            
            row = [file_name,cell,t,vgg16[k],vgg19[k],resnet50[k],inceptv2[k],inceptv3[k],dense121[k],dense169[k],dense201[k],xception[k]]
            data_writer.writerow(row)
            t = t + 1
            if t > time:
                 t = 1
                 cell = cell + 1

def prepare_data(file_name):
    csvfile = open(file_name, 'w', newline ='')
    data_writer = csv.writer(csvfile, delimiter=',')
    data_writer.writerow(['Image','Cell','Time','Status'])
    return data_writer,csvfile


def write_data(file_name,data_writer,status,time):    
        t = 1
        cell = 1
        for k in range(0,len(status)):            
            row = [file_name,cell,t,status[k]]
            data_writer.writerow(row)
            t = t + 1
            if t > time:
                 t = 1
                 cell = cell + 1


            

def get_images(whole_image, boxes,size):
    if boxes == None:
        return None
    image = np.copy(whole_image);    
    images = []
    for box in boxes:
        xmin  = int((box.x) * image.shape[1]) - int(size/2)
        xmax  = int((box.x) * image.shape[1]) + int(size/2)
        ymin  = int((box.y) * image.shape[0]) - int(size/2)
        ymax  = int((box.y) * image.shape[0]) + int(size/2)
        if xmin < 0 or xmax < 100:
            xmin = 0
            xmax = 100
        if ymin < 0 or ymax < 100:
            ymin = 0
            ymax = 100
        

        if ymax > image.shape[0]:
            ymax = image.shape[0]
            ymin = ymax - 100
        if xmax > image.shape[1]:
            xmax = image.shape[1]
            xmin = xmax - 100
        images.append(np.copy(image[ymin:ymax,xmin:xmax]))
       
        #image[ymin:ymax,xmin:xmax] = 0 #This is to elimanate duplicates
        
        
    return np.array(images)

def get_images2(whole_image, boxes,size):
    if boxes == None:
        return None
    image = np.copy(whole_image);    
    images = []
    for box in boxes:
        xmin  = int((box.x) * image.shape[2]) - int(size/2)
        xmax  = int((box.x) * image.shape[2]) + int(size/2)
        ymin  = int((box.y) * image.shape[1]) - int(size/2)
        ymax  = int((box.y) * image.shape[1]) + int(size/2)
        if xmin < 0:
               xmin = 0
               xmax = 100
        if ymin < 0:
            ymin = 0
            ymax = 100

        if ymax > image.shape[1]:
            ymax = image.shape[1]
            ymin = ymax - 100
        if xmax > image.shape[2]:
            xmax = image.shape[2]
            xmin = xmax - 100
        for num in range(0,image.shape[0]):
           images.append(np.copy(image[num,ymin:ymax,xmin:xmax]))
        #image[ymin:ymax,xmin:xmax] = 0 #This is to elimanate duplicates
        
        
    return np.array(images)
     
def get_montages(whole_image, boxes,size):
    if boxes == None:
        return None
    image = np.copy(whole_image);    
    images = []
    temp = np.zeros((size,image.shape[0] * size))
    for box in boxes:
        xmin  = int((box.x) * image.shape[2]) - int(size/2)
        xmax  = int((box.x) * image.shape[2]) + int(size/2)
        ymin  = int((box.y) * image.shape[1]) - int(size/2)
        ymax  = int((box.y) * image.shape[1]) + int(size/2)
        if xmin < 0:
               xmin = 0
               xmax = 100
        if ymin < 0:
            ymin = 0
            ymax = 100

        if ymax > image.shape[1]:
            ymax = image.shape[1]
            ymin = ymax - 100
        if xmax > image.shape[2]:
            xmax = image.shape[2]
            xmin = xmax - 100
        for num in range(0,image.shape[0]):
           temp2 = np.copy(image[num,ymin:ymax,xmin:xmax])
           #print(temp2.shape)
           #print(temp.shape)
           temp[0:size,num*size:(num+1)*size] = temp2
        images.append(np.copy(temp))
        
    return np.array(images)

def decode_netout(netout, obj_threshold, nms_threshold, anchors, nb_class):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if classes.any():
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x, y, w, h, confidence, classes)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)

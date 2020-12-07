# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:43:03 2019

@author: Eric Danielson
"""

from skimage import io
import os
import argparse
from tqdm import tqdm

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):
    
    image_path   = args.input
    files = os.listdir(image_path)
    for f in tqdm(files):
        print("Trying to open " + f)
        try:
            image = io.imread(image_path + f)            
        except:
            print("File " + f + " is corrupted")
            image = "empty"
        if image != "empty":
            if len(image.shape) < 3:
                print("File " + f + " is corrupted")

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
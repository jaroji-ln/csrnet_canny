import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')   # get gt path
    img = Image.open(img_path).convert('RGB')                                   # open image
    gt_file = h5py.File(gt_path)                                                # open gt file
    target = np.asarray(gt_file['density'])                                     # cast to array
    target = cv2.resize(target, (int(target.shape[1]/8), int(target.shape[0]/8)), interpolation=cv2.INTER_CUBIC) * 64 # resize devided by 8
    return img,target
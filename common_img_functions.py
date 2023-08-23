import os 
import numpy as np 
import cv2 # opencv
import matplotlib.pyplot as plt

def one_neg(inlist):
    return np.array([ np.array(inlist) <= 0 ]).any()

def apply_margin(pt1, pt2, margin): 
    npt1, npt2 = [0,0], [0,0]

    npt1[0] =  int(pt1[0] - margin)
    npt1[1] =  int(pt1[1] - margin)
    npt2[0] =  int(pt2[0] + margin)
    npt2[1] =  int(pt2[1] + margin)

    if one_neg(npt1)  or one_neg(npt2) :  
        return pt1, pt2

    return  npt1, npt2

def crop_image(img, cor, from_path=True, margin=None) : 

    img = cv2.imread(img) if from_path else img

    pt1  = [int(cor['xmin']), int(cor['ymin'])]
    pt2 =  [int(cor['xmax']), int(cor['ymax'])]
    
    if margin != None : 
        pt1, pt2 = apply_margin(pt1, pt2, margin)

    # crop ymin:ymax, xmin:xmax
    return img[pt1[1]:pt2[1],pt1[0]:pt2[0]]

def square_img(img, from_path=True):
    
    img = cv2.imread(img) if from_path else img
    height, width, _ = img.shape
    size_square = min(height, width)
    half_size_square = int(size_square / 2 )
    # new coordinates
    (x0, y0) = int(width/2), int(height / 2 )
    # print( (x0, y0))
    xmin = x0 - half_size_square
    xmax = x0 + half_size_square
    ymin = y0 - half_size_square
    ymax = y0 + half_size_square
    
    cor = {'xmin' : xmin,
            'xmax' : xmax,
            'ymin' : ymin,
            'ymax' : ymax
            }
    # print(cor)

    return crop_image(img, cor, from_path=False)
    
    
    
def resize_image(img, new_dim, from_path=True):

    img = cv2.imread(img) if from_path else img
    
    return cv2.resize(img, new_dim, interpolation = cv2.INTER_AREA)

def read_img(img_path) : 
    return cv2.imread(img_path)
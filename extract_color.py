#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:31:06 2022

@author: gw
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import pyramid_reduce, resize
import cv2
import os, glob





def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def color_extract(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_COLOR).astype(np.float32)
    #img = img.reshape(512,512,3,1) # 512X512X3X1
    Red = img[:,:,0]
    print(np.max(Red))
    for i in range(512):
        for j in range(512):
            if Red[i][j] < 200.0:
                Red[i][j]=0   
   
    print(Red.shape)
    '''
    img = np.expand_dims(img, axis=-1)
    img = np.delete(img,(2), axis = -2)
    img = np.delete(img,(1), axis = -2)
    img = np.squeeze(img , axis = -1)
    '''
    #plt.imshow(img([:,:,0])
    #plt.show()
    img = cv2.imwrite("repd2fff.jpg", Red)
    #print(img.shape)


if __name__ == "__main__":
    """dataset_path"""
    dataset_path = "archive/masks/masks"
    img = glob.glob(os.path.join(dataset_path,'*.jpg'))
    print(img)
    #for s in range(img)
    #    color_extract("")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
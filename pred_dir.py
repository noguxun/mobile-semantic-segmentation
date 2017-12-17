from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from keras.models import load_model
from keras.utils import CustomObjectScope
from sklearn.model_selection import train_test_split
from scipy.misc import imresize
from scipy.ndimage import imread

from data import seed, standardize
from loss import np_dice_coef
from nets.MobileUNet import custom_objects
from glob import glob
import re
from nets.MobileUNet import *

SAVED_MODEL1 = 'artifacts/model.h5'

img_size = 128

def beautify(pred):
    return pred
    
    row = pred.shape[0]
    col = pred.shape[1]
   
    for i in range(0, row):
        for j in range(0, col):
            val = pred[i][j]
            if val <= 0.002:
                val = 0
            elif val > 0.002 and val < 0.2:
                val = 0.5
            elif val >= 0.2:
                val = 1
            pred[i][j] = val
    '''
    for i in range(0, row):
        for j in range(0, col):
            val = pred[i][j]
            if val < 0.00001:
                val = 0
            elif val < 0.0001:
                val = 0.1 + val * 1000
            elif val < 0.001:
                val = 0.2 + val * 100 
            elif val < 0.01:
                val = 0.3 + val * 10
            elif val < 0.1:
                val = 0.4 + val
            elif val < 0.2:
                val = 0.5 + val
            elif val < 0.5:
                val = 0.5 + val
            else:
                val = val

            pred[i][j] = val
    '''        
    
    return pred
    
def main(img_dir):
    if True:
        with CustomObjectScope(custom_objects()):
            model = load_model(SAVED_MODEL1)
    
    else:
        weight_file = 'artifacts/checkpoint_weights.73-0.01.h5'
        model = MobileUNet(input_shape=(128, 128, 3),
                       alpha=1,
                       alpha_up=0.25)

    
        model.load_weights(weight_file, by_name=True)
        
    model.summary()
    

    img_files = glob(img_dir + '/*.jpg')
    img_files = sorted(img_files, key=os.path.getmtime)

    for img_file in img_files: #reversed(img_files):
        img = imread(img_file)
        img = imresize(img, (img_size, img_size))

        mask_file = re.sub('img2.jpg$', 'msk1.png',img_file)

        mask = imread(mask_file)
       
        mask = imresize(mask, (img_size, img_size), interp='nearest')
        mask1 = mask[:,:,0] 
        
        batched1 = img.reshape(1, img_size, img_size, 3).astype(float)
        pred1 = model.predict(standardize(batched1)).reshape(img_size, img_size)
      
        mask1 = mask1.astype(float) / 255 
        
        
        dice = np_dice_coef(mask1, pred1)
        
        print('dice1: ', dice)
        print('sum ' ,np.sum(mask1))
        print('shap ', mask1.shape)

        pred1 = beautify(pred1)
        
        if True:
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.subplot(2, 2, 2)
            plt.imshow(mask)
            plt.subplot(2, 2, 3)
            plt.imshow(pred1)
            plt.subplot(2, 2, 4)
            plt.imshow(mask1)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_dir',
        type=str,
        default='data/photo_bought',
    )

    args, _ = parser.parse_known_args()
    main(**vars(args))

from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import argparse

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

SAVED_MODEL1 = 'artifacts/model.h5'

img_size = 128

def beautify(pred):
    row = pred.shape[0]
    col = pred.shape[1]
    
    for i in range(0, row):
        for j in range(0, col):
            val = pred[i][j] * 200 
            if val > 1:
                val = 1
            pred[i][j] = val
    
    return pred
    
def main(img_dir):
    with CustomObjectScope(custom_objects()):
        model1 = load_model(SAVED_MODEL1)
        
    img_files = glob(img_dir + '/*.jpg')

    for img_file in img_files:
        img = imread(img_file)
        img = imresize(img, (img_size, img_size))

        mask_file = re.sub('img2.jpg$', 'msk1.png',img_file)

        mask = imread(mask_file)
       
        mask = imresize(mask, (img_size, img_size), interp='nearest')
        mask = mask[:,:,0] 
        
        batched1 = img.reshape(1, img_size, img_size, 3).astype(float)
        pred1 = model1.predict(standardize(batched1)).reshape(img_size, img_size)
      
        mask1 = mask.astype(float) / 255 
        
        
        import pdb
        pdb.set_trace()
        
        
        dice = np_dice_coef(mask1, pred1)
        print('dice1: ', dice)

        pred1 = beautify(pred1)
        
        if True:
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.subplot(2, 2, 2)
            plt.imshow(pred1)
            plt.subplot(2, 2, 3)
            plt.imshow(mask)
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

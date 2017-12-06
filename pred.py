from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope
from sklearn.model_selection import train_test_split

from data import seed, standardize
from loss import np_dice_coef
from nets.MobileUNet import custom_objects

SAVED_MODEL1 = 'artifacts/model.h5'

size = 128


def main():
    with CustomObjectScope(custom_objects()):
        model1 = load_model(SAVED_MODEL1)

    images = np.load('data/id_pack/images-128.npy')
    masks = np.load('data/id_pack/masks-128.npy')
    # only hair
    masks = masks[:, :, :, 0].reshape(-1, size, size)

    _, images, _, masks = train_test_split(images,
                                           masks,
                                           test_size=0.2,
                                           random_state=seed)

    for img, mask in zip(images, masks):
        batched1 = img.reshape(1, size, size, 3).astype(float)
        batched2 = img.reshape(1, size, size, 3).astype(float)

        t1 = time.time()
        pred1 = model1.predict(standardize(batched1)).reshape(size, size)
        elapsed = time.time() - t1
        print('elapsed1: ', elapsed)

        dice = np_dice_coef(mask.astype(float) / 255, pred1)
        print('dice1: ', dice)
        
        print(pred1.shape)


        if True:
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.subplot(2, 2, 2)
            plt.imshow(mask)
            plt.subplot(2, 2, 3)
            plt.imshow(pred1)
            plt.show()

        #break


if __name__ == '__main__':
    main()

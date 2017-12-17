import argparse
import os
import re
import numpy as np

from glob import glob

#from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize
from scipy.ndimage import imread
from sklearn.model_selection import train_test_split


img_size = 256
data_dir = "data/id_images"
out_dir = "data/id_pack"
img_array = []
mask_array = []

img_files = glob(data_dir + '/*.jpg')
for img_path in img_files:
    msk_path = re.sub('img2.jpg$', 'msk1.png',img_path)
    print(msk_path)
    print(img_path)
    img = imread(img_path, mode='RGB')
    mask = imread(msk_path, mode='RGB')

    img = imresize(img, (img_size, img_size))
    mask = imresize(mask, (img_size, img_size), interp='nearest')

    img_array.append(img)
    mask_array.append(mask)

    if False:
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.show()

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

np.save(out_dir + '/images-{}.npy'.format(img_size), np.array(img_array))
np.save(out_dir + '/masks-{}.npy'.format(img_size), np.array(mask_array))

print("Total processed " + str(len(img_files)))

def create_data(data_dir, out_dir, img_size):
    """
    It expects following directory layout in data_dir.

    images/
      0001.jpg
      0002.jpg
    masks/
      0001.ppm
      0002.ppm

    Mask image has 3 colors R, G and B. R is hair. G is face. B is bg.
    Finally, it will create images.npy and masks.npy in out_dir.

    :param data_dir:
    :param out_dir:
    :param img_size:
    :return:
    """
    """
    img_files = glob(data_dir + '/images/*.jpg')
    mask_files = glob(data_dir + '/masks/*.ppm')
    X = []
    Y = []
    for img_path, mask_path in zip(img_files, mask_files):
        img = imread(img_path)
        img = imresize(img, (img_size, img_size))

        mask = imread(mask_path)
        mask = imresize(mask, (img_size, img_size), interp='nearest')

        # debug
        if False:
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.show()

        X.append(img)
        Y.append(mask)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.save(out_dir + '/images-{}.npy'.format(img_size), np.array(X))
    np.save(out_dir + '/masks-{}.npy'.format(img_size), np.array(Y))


    ))
    """

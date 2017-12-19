import argparse
import os
import re
import numpy as np

from glob import glob

#from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize
from scipy.ndimage import imread
from sklearn.model_selection import train_test_split

def main(img_size):
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_size',
        type=int,
        default=192,
    )
    args, _ = parser.parse_known_args()
    main(**vars(args))

from __future__ import print_function

import numpy as np
from keras.utils import plot_model

import argparse
import os

from keras import callbacks, optimizers

from data import load_data
from learning_rate import create_lr_schedule
from loss import dice_coef_loss, dice_coef, recall, precision
from nets.MobileUNet import MobileUNet
from nets.MobileUNet import custom_objects
from keras.utils import CustomObjectScope
from keras.models import load_model
from keras import backend as K

from nets.SqueezeNet import SqueezeNet

img_height = img_width = 256

model_MobileUNet = MobileUNet(input_shape=(img_height, img_width, 3), alpha=1, alpha_up=0.25)

model_MobileUNet.summary()
plot_model(model_MobileUNet, to_file="mobile_u_net_model.png", show_shapes=True)

#model_SqueezeNet = SqueezeNet(input_shape=(img_height, img_width, 3), classes=(img_height*img_width))
#plot_model(model_SqueezeNet, to_file="squeeze_net_model.png", show_shapes=True)


#model_SqueezeNet_notop = SqueezeNet(input_shape=(img_height, img_width, 3), include_top=False, pooling='max')
#plot_model(model_SqueezeNet_notop, to_file="squeeze_net_no_top_model.png", show_shapes=True)
'''
img = image.load_img('../images/cat.jpeg', target_size=(227, 227))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
'''

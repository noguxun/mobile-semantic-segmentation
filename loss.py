import numpy as np
from keras import backend as K
import tensorflow as tf

smooth = 1e-5

def loss_gu(y_true, y_pred):
    # we want background to have more loss
    adj = y_true + 1  # element 2: backgroud, element 1: forground
    diff = y_pred - y_true
    diff_adj = tf.multiply(diff, adj)
    return K.mean(K.abs(diff_adj), axis=-1)

def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_f, 0, 1)))

    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))

    return true_positives / (possible_positives + K.epsilon())


def f1_score(y_true, y_pred):
    return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def np_dice_coef(y_true, y_pred):
    tr = y_true.flatten()
    pr = y_pred.flatten()
    
    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)

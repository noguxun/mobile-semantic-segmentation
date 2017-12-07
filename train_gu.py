from __future__ import print_function

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
import tensorflow as tf

checkpoint_path = 'artifacts/checkpoint_weights.{epoch:02d}-{val_loss:.2f}.h5'
trained_model_path = 'artifacts/model.h5'
SAVED_MODEL1 = 'artifacts/model_transfer.h5'



def train(img_file, mask_file, epochs, batch_size):
    train_gen, validation_gen, img_shape, train_len, val_len = load_data(img_file, mask_file)

    img_height = img_shape[0]
    img_width = img_shape[1]
    lr_base = 0.01 * (float(batch_size) / 16)

    if False:
        with CustomObjectScope(custom_objects()):
            model = load_model(SAVED_MODEL1)
    
    else:
        model = MobileUNet(input_shape=(img_height, img_width, 3),
                        alpha=1,
                        alpha_up=0.25)

    model.summary()
    model.compile(
        optimizer=optimizers.SGD(lr=0.00001, momentum=0.9),
        # optimizer=Adam(lr=0.001),
        # optimizer=optimizers.RMSprop(),
        #loss=dice_coef_loss,
        # loss='mean_absolute_error',
        loss = loss_gu,
        metrics=[
            dice_coef,
            recall,
            precision,
            loss_gu,
        ],
    )

    # callbacks
    scheduler = callbacks.LearningRateScheduler(
        create_lr_schedule(epochs, lr_base=lr_base, mode='progressive_drops'))
    tensorboard = callbacks.TensorBoard(log_dir='./logs')
    csv_logger = callbacks.CSVLogger('logs/training.csv')
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           save_best_only=True)
    
    '''
        looks like we do have some legacy support issue
        the steps_per_epoch and validation_steps is actually the number of sample
        
        legacy_generator_methods_support = generate_legacy_method_interface(
            allowed_positional_args=['generator', 'steps_per_epoch', 'epochs'],
            conversions=[('samples_per_epoch', 'steps_per_epoch'),
                        ('val_samples', 'steps'),
                        ('nb_epoch', 'epochs'),
        ('nb_val_samples', 'validation_steps'),
    '''
    nb_train_samples = train_len 
    nb_validation_samples = val_len
   
    print("training sample is " + str(nb_train_samples))

    model.fit_generator(
        generator=train_gen(),
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_gen(),
        validation_steps=nb_validation_samples // batch_size,
        # callbacks=[tensorboard, checkpoint, csv_logger],
        #callbacks=[scheduler, tensorboard, checkpoint, csv_logger],
    )

    model.save(trained_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_file',
        type=str,
        default='data/id_pack/images-128.npy',
        help='image file as numpy format'
    )
    parser.add_argument(
        '--mask_file',
        type=str,
        default='data/id_pack/masks-128.npy',
        help='mask file as numpy format'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=250,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
    )
    args, _ = parser.parse_known_args()

    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    train(**vars(args))

# coding=utf-8
import json

import numpy as np
from keras import metrics
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from linc_cv import CLASS_INDICIES_PATH, WHISKER_MODEL_PATH
from linc_cv import datapath
from linc_cv.ml import ClassifierError
from linc_cv.whiskers.sgdr import SGDRScheduler
from linc_cv.whiskers.utils import get_class_weights


def train_whiskers(validation, epochs):
    """
    Train a neural network to perform unique whisker pattern identification.
    If validation is False, epochs must be set to avoid overtraining.
    For example, after training and validating the neural network's performance
    on a train/test split, you can use the epoch with the lowest val_loss as a
    starting point for retraining the neural network from scratch on the entire
    dataset.
    """

    if not validation and not epochs:
        raise ClassifierError('if no validation, epochs must be provided')

    print(f'training whisker classifier with validation={validation} for epochs={epochs}')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.,
        rotation_range=25,
        zoom_range=0.15,
        height_shift_range=0.15,
        width_shift_range=0.15,
        fill_mode='nearest',
        samplewise_center=True,
        samplewise_std_normalization=True, )

    batch_size = 28

    if validation:
        trainpath = ['whiskers_images_traintest', 'train']
    else:
        # train on all whisker images
        trainpath = ['whiskers_images_filtered']

    train_generator = train_datagen.flow_from_directory(
        datapath(trainpath),
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = None
    if validation:
        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            samplewise_center=True,
            samplewise_std_normalization=True, )

        validation_generator = test_datagen.flow_from_directory(
            datapath(['whiskers_images_traintest', 'test']),
            target_size=(299, 299,),
            batch_size=batch_size,
            class_mode='categorical')

        assert train_generator.num_classes == validation_generator.num_classes, (
            train_generator.num_classes, validation_generator.num_classes,)
        y = np.concatenate((train_generator.classes, validation_generator.classes,))
    else:
        y = train_generator.classes

    with open(CLASS_INDICIES_PATH, 'w') as f:
        json.dump(train_generator.class_indices, f)
    num_classes = train_generator.num_classes
    max_lr = 1e-1
    min_lr = 1e-6
    epoch_size = 300
    class_weight = get_class_weights(y)
    model = InceptionResNetV2(weights=None, classes=num_classes)
    optimizer = SGD(lr=max_lr, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[metrics.categorical_accuracy])
    lrs = SGDRScheduler(
        min_lr=min_lr,
        max_lr=max_lr,
        steps_per_epoch=epoch_size,
        lr_decay=0.8,
        cycle_length=1,
        mult_factor=1.0)
    if validation:
        save_best_only = True
    else:
        save_best_only = False
    mcp = ModelCheckpoint(
        filepath=WHISKER_MODEL_PATH, verbose=1,
        save_best_only=save_best_only)
    if validation:
        model.fit_generator(
            train_generator,
            steps_per_epoch=epoch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=epoch_size,
            max_queue_size=512,
            use_multiprocessing=True,
            workers=8,
            class_weight=class_weight,
            callbacks=[lrs, mcp])
    else:
        model.fit_generator(
            train_generator,
            steps_per_epoch=epoch_size,
            epochs=epochs,
            max_queue_size=512,
            use_multiprocessing=True,
            workers=8,
            class_weight=class_weight,
            callbacks=[lrs, mcp])

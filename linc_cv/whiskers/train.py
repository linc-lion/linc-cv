import json

import numpy as np
from keras import metrics
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from linc_cv.whiskers.sgdr import SGDRScheduler
from linc_cv.whiskers.utils import get_class_weights

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.,
    rotation_range=15,
    zoom_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
    fill_mode='nearest',
    samplewise_center=True,
    samplewise_std_normalization=True, )

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    samplewise_center=True,
    samplewise_std_normalization=True, )

epoch_size = 300
batch_size = 28

train_generator = train_datagen.flow_from_directory(
    'whiskers_images_traintest/train',
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'whiskers_images_traintest/test',
    target_size=(299, 299,),
    batch_size=batch_size,
    class_mode='categorical')

assert train_generator.num_classes == validation_generator.num_classes, (
    train_generator.num_classes, validation_generator.num_classes,)
with open('data/whiskers/class_indicies.json', 'w') as f:
    json.dump(validation_generator.class_indices, f)
num_classes = train_generator.num_classes

max_lr = 1e-1
min_lr = 1e-6

y = np.concatenate((train_generator.classes, validation_generator.classes,))
class_weight = get_class_weights(y)

try:
    model = load_model('whiskers.h5')
except OSError:
    model = InceptionResNetV2(weights=None, classes=num_classes)
    optimizer = SGD(lr=max_lr, momentum=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[metrics.categorical_accuracy])
lrs = SGDRScheduler(
    min_lr=min_lr,
    max_lr=max_lr,
    steps_per_epoch=epoch_size,
    lr_decay=0.8,
    cycle_length=1,
    mult_factor=2.0)
mcp = ModelCheckpoint(filepath='whiskers.h5', verbose=1, save_best_only=True)
model.fit_generator(
    train_generator,
    steps_per_epoch=epoch_size,
    epochs=5000,
    validation_data=validation_generator,
    validation_steps=epoch_size,
    max_queue_size=512,
    use_multiprocessing=True,
    workers=8,
    class_weight=class_weight,
    callbacks=[lrs, mcp])

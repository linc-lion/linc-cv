# coding=utf-8

# coding=utf-8
import os
import random
import shutil
from collections import Counter
from multiprocessing import cpu_count

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from linc_cv import WHISKER_IMAGES_PATH, WHISKER_IMAGES_TRAINTEST_PATH
from linc_cv import WHISKER_MODEL_PATH
from linc_cv.whiskers.sgdr import SGDRScheduler
from linc_cv.whiskers.utils import get_class_weights


def train_whiskers():
    input_shape = (299, 299, 3,)
    batch_size = 20
    XY = []
    for root, dirs, files in os.walk(WHISKER_IMAGES_PATH):
        for f in files:
            path = os.path.join(root, f)
            label = path.split(os.path.sep)[-2]
            XY.append([path, label])
    assert XY
    random.shuffle(XY)
    X, y = zip(*XY)
    valid_labels = set(label for label, count in Counter(y).items() if count > 2)
    Xp = []
    yp = []
    for x_, y_ in zip(X, y):
        if y_ in valid_labels:
            Xp.append(x_)
            yp.append(y_)
    X_train, X_test, y_train, y_test = train_test_split(Xp, yp, stratify=yp)

    try:
        shutil.rmtree(WHISKER_IMAGES_TRAINTEST_PATH)
    except FileNotFoundError:
        pass

    for mode in ('train', 'test',):
        for x, y in zip(eval(f'X_{mode}'), eval(f'y_{mode}')):
            np = os.path.join(WHISKER_IMAGES_TRAINTEST_PATH, mode, y, os.path.basename(x) + '.jpg')
            os.makedirs(os.path.dirname(np), exist_ok=True)
            shutil.copyfile(x, np)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        rotation_range=5)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(WHISKER_IMAGES_TRAINTEST_PATH, 'train'),
        target_size=input_shape[:-1],
        batch_size=batch_size,
        class_mode='categorical')

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True, )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(WHISKER_IMAGES_TRAINTEST_PATH, 'test'),
        target_size=input_shape[:-1],
        batch_size=batch_size,
        class_mode='categorical')

    assert train_generator.num_classes == validation_generator.num_classes
    num_classes = validation_generator.num_classes

    imagenet_weights_path = os.path.expanduser(
        '~/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(imagenet_weights_path, by_name=True)
    max_lr = 0.01000
    min_lr = 0.00001
    model.compile(
        optimizer=SGD(lr=max_lr, momentum=0.8),
        loss='categorical_crossentropy', metrics=['accuracy'])
    class_weights = get_class_weights(train_generator.classes)
    mc = ModelCheckpoint(WHISKER_MODEL_PATH, save_best_only=True, verbose=1)
    lrs = SGDRScheduler(
        max_lr=max_lr,
        min_lr=min_lr,
        steps_per_epoch=len(X_train) // batch_size,
        lr_decay=1,
        cycle_length=5,
        mult_factor=1)
    model.fit_generator(
        train_generator, epochs=50000,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(X_test) // batch_size,
        use_multiprocessing=True,
        workers=cpu_count(),
        class_weight=class_weights,
        callbacks=[mc, lrs])

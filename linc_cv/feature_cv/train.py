# coding=utf-8
import os
import random
import shutil
from collections import Counter
from multiprocessing import cpu_count

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from linc_cv import CV_IMAGES_PATH, CV_IMAGES_TRAINTEST_PATH


def train_cv():
    input_shape = (299, 299, 3,)
    batch_size = 24
    XY = []
    for root, dirs, files in os.walk(CV_IMAGES_PATH):
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
        shutil.rmtree(CV_IMAGES_TRAINTEST_PATH)
    except FileNotFoundError:
        pass

    for mode in ('train', 'test',):
        for x, y in zip(eval(f'X_{mode}'), eval(f'y_{mode}')):
            np = os.path.join(CV_IMAGES_TRAINTEST_PATH, mode, y, os.path.basename(x) + '.jpg')
            os.makedirs(os.path.dirname(np), exist_ok=True)
            shutil.copyfile(x, np)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(CV_IMAGES_TRAINTEST_PATH, 'train'),
        target_size=input_shape[:-1],
        batch_size=batch_size,
        class_mode='categorical')

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True, )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(CV_IMAGES_TRAINTEST_PATH, 'test'),
        target_size=input_shape[:-1],
        batch_size=batch_size,
        class_mode='categorical')

    assert train_generator.num_classes == validation_generator.num_classes
    num_classes = validation_generator.num_classes

    imagenet_weights_path = os.path.expanduser(
        '~/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(imagenet_weights_path, by_name=True)
    # freeze all but last 2 layers
    for layer in model.layers[:-2]:
        layer.trainable = False
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(
        train_generator, epochs=500,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(X_test) // batch_size,
        use_multiprocessing=True, workers=cpu_count())

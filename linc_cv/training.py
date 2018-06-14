import json
import os
import shutil
from collections.__init__ import Counter
from multiprocessing import cpu_count

from keras import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from sklearn.model_selection import train_test_split

from linc_cv import get_class_weights, INPUT_SHAPE
from linc_cv.sgdr import SGDRScheduler


def preprocess_input_new(x):
    img = preprocess_input(img_to_array(x))
    return array_to_img(img)


def train(*, images_dir, images_traintest_dir, lut_path, model_path,
          training_idg_params, testing_idg_params, training_log):
    batch_size = 20
    X = []
    y = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            path = os.path.join(root, f)
            label = path.split(os.path.sep)[-2]
            X.append(path)
            y.append(label)
    valid_labels = set(label for label, count in Counter(y).items() if count > 2)
    Xp = []
    yp = []
    for x_, y_ in zip(X, y):
        if y_ in valid_labels:
            Xp.append(x_)
            yp.append(y_)
    X_train, X_test, y_train, y_test = train_test_split(Xp, yp, stratify=yp, shuffle=True)

    try:
        shutil.rmtree(images_traintest_dir)
    except FileNotFoundError:
        pass

    for mode in ('train', 'test',):
        for x, y in zip(eval(f'X_{mode}'), eval(f'y_{mode}')):
            np = os.path.join(images_traintest_dir, mode, y, os.path.basename(x) + '.jpg')
            os.makedirs(os.path.dirname(np), exist_ok=True)
            os.symlink(x, np)

    train_datagen = ImageDataGenerator(**{
        **training_idg_params,
        'preprocessing_function': preprocess_input_new})
    train_generator = train_datagen.flow_from_directory(
        os.path.join(images_traintest_dir, 'train'),
        target_size=INPUT_SHAPE[:-1],
        batch_size=batch_size,
        class_mode='categorical',
        follow_links=True)

    test_datagen = ImageDataGenerator(**{
        **testing_idg_params,
        'preprocessing_function': preprocess_input_new})

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(images_traintest_dir, 'test'),
        target_size=INPUT_SHAPE[:-1],
        batch_size=batch_size,
        class_mode='categorical',
        follow_links=True)

    assert train_generator.num_classes == validation_generator.num_classes
    num_classes = validation_generator.num_classes

    with open(lut_path, 'w') as f:
        json.dump(validation_generator.class_indices, f)

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
    mc = ModelCheckpoint(model_path, save_best_only=True, verbose=1)
    lrs = SGDRScheduler(
        max_lr=max_lr,
        min_lr=min_lr,
        steps_per_epoch=len(X_train) // batch_size,
        lr_decay=1,
        cycle_length=5,
        mult_factor=1)
    csvl = CSVLogger(training_log)
    model.fit_generator(
        train_generator, epochs=100,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(X_test) // batch_size,
        use_multiprocessing=True,
        workers=cpu_count(),
        class_weight=class_weights,
        callbacks=[csvl, mc, lrs])

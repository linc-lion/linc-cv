import json
import multiprocessing
import shutil
import sys
import tempfile
from collections import defaultdict
from operator import itemgetter

import numpy as np
import requests
import scipy.cluster.hierarchy
import skimage.color
import skimage.draw
import skimage.exposure
import skimage.feature
import skimage.transform
import skimage.util
import tables as tb
from keras import metrics
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.callbacks import EarlyStopping
from keras.layers import GlobalAveragePooling2D, \
    Dropout, Reshape, Activation, Conv2D, BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

linc_features = None
features_lut = None
model = None


class ClassifierError(Exception):
    @property
    def message(self):
        return self.args[0]


def download_image(image_url):
    with tempfile.NamedTemporaryFile() as t:
        r = requests.get(image_url, stream=True)
        shutil.copyfileobj(r.raw, t)
        t.flush()
        t.seek(0)
        try:
            from PIL import Image
            with Image.open(t.name) as im:
                img = im.resize((224, 224,), resample=Image.LANCZOS).convert('RGB')
            return img
        except OSError:
            raise ClassifierError(f'unable to open image: {image_url}')


def initialize():
    global linc_features
    global features_lut
    global model
    if model is None:
        model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3,))
        print('initialized model')
    if linc_features is None:
        linc_features = tb.open_file('data/linc_features.h5').root._v_children
        print('initialized linc_features')
    if features_lut is None:
        with open('data/features_lut.json') as f:
            features_lut = json.load(f)
        print('initialized features_lut')
    return linc_features, features_lut, model


def whisker_image_to_mask(img):
    img_arr = np.array(img)
    img = skimage.color.rgb2gray(img_arr)
    img = skimage.exposure.equalize_adapthist(img)
    img = skimage.util.invert(img)
    blobs = skimage.feature.blob_log(img, min_sigma=3, max_sigma=6, num_sigma=50)
    try:
        clusters = scipy.cluster.hierarchy.fclusterdata(blobs[:, :-1], 40, criterion='distance')
    except ValueError:
        raise ClassifierError('unable to cluster whisker point data')
    major_cluster = np.bincount(clusters).argmax()
    blobs = blobs[np.where(clusters == major_cluster)]
    mask = np.zeros(img_arr.shape)
    if mask.shape != (224, 224, 3,):
        raise ClassifierError('mask shape is invalid')
    for r, c, sigma in blobs:
        rr, cc = skimage.draw.circle(r, c, sigma, shape=img.shape)
        mask[rr, cc] = 1
    return mask


def extract_general_image_features(img):
    global linc_features, features_lut, model
    linc_features, features_lut, model = initialize()
    try:
        x = image.img_to_array(img)
    except ValueError:
        raise ClassifierError('unable to extract features from image')
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features


def image_urls_to_features(data):
    image_index, image_url, feature_type = data
    img = download_image(image_url)
    features = extract_general_image_features(img)
    return image_index, features, feature_type


def generate_linc_lut():
    global linc_features
    with open('data/images_lut.json') as f:
        linc_images_lut = json.load(f)
    features_lut = defaultdict(lambda: defaultdict(list))
    image_index = defaultdict(lambda: 0)
    d = []
    for lion_id, feature_types in linc_images_lut.items():
        for feature_type, image_urls in feature_types.items():
            for image_url in image_urls:
                d.append((image_index[feature_type], image_url, feature_type,))
                features_lut[feature_type][lion_id].append(image_index[feature_type])
                image_index[feature_type] += 1
    with open('data/features_lut.json', 'w') as f:
        json.dump(features_lut, f)
    cmp = tb.Filters(complib='blosc', complevel=9, fletcher32=True, bitshuffle=True, least_significant_digit=3)
    f = tb.open_file('linc_features.h5', mode='w', title="LINC Neural Network Extracted Features", filters=cmp)
    linc_features = f.root._v_children
    for feature_type in features_lut:
        shape = (image_index[feature_type], 7, 7, 1024,)
        f.create_carray(f.root, feature_type, tb.Float32Atom(), shape=shape)
    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(
            processes=processes) as pool:
        for image_index, image_features, feature_type in pool.imap_unordered(image_urls_to_features, d):
            if image_features is None:
                continue
            linc_features[feature_type][image_index] = image_features
            sys.stdout.write('.')
            sys.stdout.flush()
    f.close()


def sort_lion_xy(x, y):
    # hdf5 requires sorted ascending array indicies
    s_xy = sorted(zip(x, y), key=itemgetter(0))

    # remove duplicates
    xf_s = set()
    xf = []
    yf = []
    for x, y in s_xy:
        if x in xf_s:
            continue
        xf_s.add(x)
        xf.append(x)
        yf.append(y)
    return xf, yf


def lions_to_xy_val(feature_type, lion_ids,
                    test_split_factor=0.8,
                    test_feature_idx=None,
                    gt_class=None):
    global linc_features, features_lut, model
    linc_features, features_lut, model = initialize()
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    if not all(lion_id in features_lut[feature_type].keys() for lion_id in lion_ids):
        raise ClassifierError('at least one requested lion_id to search against is '
                              'not present in the existing lion_ids precomputed '
                              'features database')
    for lion_id in list(set(lion_ids)):
        image_ids = features_lut[feature_type][lion_id]
        if lion_id == gt_class:
            split_idx = int(len(image_ids) * test_split_factor)
            train_idxs = image_ids[:split_idx]
            test_idxs = image_ids[split_idx:]
            if set(X_train).intersection(train_idxs) != set():
                raise ClassifierError('attempted to incorporate training ')
            for train_idx in train_idxs:
                if train_idx == test_feature_idx:
                    # print('skipping: avoided including test feature, gt_class & train')
                    continue
                X_train.append(train_idx)
                y_train.append(lion_id)
            for test_idx in test_idxs:
                if test_idx == test_feature_idx:
                    # print('skipping: avoided including test feature, gt_class & test')
                    continue
                X_test.append(test_idx)
                y_test.append(lion_id)
        else:
            for image_id in image_ids:
                if image_id == test_feature_idx:
                    # print('skipping: avoided including test feature, !gt_class')
                    continue
                X_train.append(image_id)
                y_train.append(lion_id)
    if not X_train:
        raise ClassifierError('training dataset is empty')
    if not X_test:
        if y_test:
            raise ClassifierError('testing dataset is empty but labels for it exist')
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
    X_train, y_train = sort_lion_xy(X_train, y_train)
    X_test, y_test = sort_lion_xy(X_test, y_test)
    if set(X_train).intersection(X_test) != set():
        raise ClassifierError('elements of the training dataset exist in the testing dataset')
    X_train = linc_features[feature_type][X_train, :]
    X_test = linc_features[feature_type][X_test, :]
    lb = LabelEncoder().fit(y_train + y_test)
    labels = lb.classes_
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    y_train = to_categorical(y_train, num_classes=len(labels))
    y_test = to_categorical(y_test, num_classes=len(labels))
    if y_train.shape[-1] != y_test.shape[-1]:
        raise ClassifierError('labels of the training dataset have a different shape than that of the testing dataset')
    nb_classes = y_train.shape[-1]
    if nb_classes < 2:
        return ClassifierError('cannot train a classifier to discriminate between < 2 classes')
    return labels, nb_classes, X_train, X_test, y_train, y_test


def feature_generic_classifier_model(num_classes):
    alpha = 1.0
    dropout = 0.1
    shape = (1, 1, int(1024 * alpha))
    model = Sequential([
        GlobalAveragePooling2D(input_shape=(7, 7, 1024,)),
        Reshape(shape, name='reshape_1'),
        Dropout(dropout, name='dropout'),
        Conv2D(num_classes, (1, 1),
               padding='same', name='conv_preds'),
        BatchNormalization(),
        Activation('softmax', name='act_softmax'),
        Reshape((num_classes,), name='reshape_2'), ])
    optimizer = SGD(lr=1e-2, decay=1e-2, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[metrics.mae, metrics.categorical_accuracy])
    return model


def classify_val(test_features, nb_classes,
                 X_train, X_test, y_train, y_test):
    if len(X_train.shape) != 4:
        raise ClassifierError('training dataset is of an invalid shape')
    if len(X_test.shape) != 4:
        raise ClassifierError('testing dataset is of an invalid shape')
    classifier_verbose = False
    feature_model = feature_generic_classifier_model(nb_classes)
    es = EarlyStopping(patience=40, verbose=int(classifier_verbose))
    feature_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=1000, callbacks=[es], verbose=int(classifier_verbose))
    loss, mae, val_acc = feature_model.evaluate(X_test, y_test)  # see model.compile metrics
    feature_model = feature_generic_classifier_model(nb_classes)
    feature_model.fit(
        np.concatenate((X_train, X_test,)),
        np.concatenate((y_train, y_test,)),
        epochs=es.stopped_epoch - es.patience, verbose=int(classifier_verbose))
    probas = feature_model.predict_proba(test_features)
    return val_acc, probas


def test_lion(feature_type, lion_ids, gt_class=None, test_feature_idx=None, test_image_url=None):
    global linc_features, features_lut, model
    linc_features, features_lut, model = initialize()
    if test_feature_idx and test_image_url:
        raise ClassifierError('cannot simultaneously process both '
                              'test_feature_idx and test_image_url')
    labels, nb_classes, X_train, X_test, y_train, y_test = \
        lions_to_xy_val(
            feature_type, lion_ids,
            test_feature_idx=test_feature_idx, gt_class=gt_class)
    if test_image_url:
        img = download_image(test_image_url)
        test_features = extract_general_image_features(img)
    else:
        test_features = linc_features[feature_type][test_feature_idx]
        test_features = np.expand_dims(test_features, 0)
    if test_features is None:
        raise ClassifierError('could not extract test features '
                              'from given lion image url or feature database')
    val_acc, probas = classify_val(
        test_features, nb_classes, X_train, X_test, y_train, y_test)
    votes = np.argmax(probas, axis=1)
    majority_vote_label = labels[np.bincount(votes).argmax()]
    print(f'majority vote: {majority_vote_label}, votes: {labels[np.argmax(probas, axis=1)]}')

    print(f'feature: {feature_type}, classifier: {val_acc}')
    correct = None
    if gt_class is not None:
        correct = str(majority_vote_label) == str(gt_class)
        print(f'correct? {correct}, pred_class == gt_class: {majority_vote_label} == {gt_class}')
    return feature_type, correct, val_acc, probas, labels


if __name__ == '__main__':
    generate_linc_lut()
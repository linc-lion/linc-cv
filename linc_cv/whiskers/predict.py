# coding=utf-8
import json
import time
from operator import itemgetter
from typing import Iterable

import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_sauvola, gaussian

from linc_cv import CLASS_INDICIES_PATH, WHISKER_MODEL_PATH
from linc_cv.ml import download_image, ClassifierError

model = None
class_indicies = None
labels = None
test_datagen = None


def initialize():
    global model
    global test_datagen
    global class_indicies
    global labels
    if class_indicies is None:
        try:
            with open(CLASS_INDICIES_PATH) as f:
                class_indicies = json.load(f)
            labels = [x[0] for x in sorted(class_indicies.items(), key=itemgetter(1))]
        except FileNotFoundError:
            pass
    if model is None:
        model = load_model(WHISKER_MODEL_PATH)
        print('loaded existing whisker model')
    if test_datagen is None:
        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            samplewise_center=True,
            samplewise_std_normalization=True, )
    return model, test_datagen, class_indicies, labels


def predict_whisker_from_preprocessed_image(path):
    model, test_datagen, class_indicies, labels = initialize()
    if class_indicies is None:
        raise ClassifierError('is whisker network trained? could not find class indicies json')
    start_time = time.time()
    im = Image.open(path).convert('RGB')
    im = im.resize((299, 299,), resample=Image.LANCZOS)
    im = img_to_array(im)
    im = np.expand_dims(im, 0)
    assert im.shape == (1, 299, 299, 3,), im.shape
    gt_label = path.split('/')[-2]
    y = np.zeros((1, len(labels),))
    y[0][class_indicies[gt_label]] = 1
    X = next(test_datagen.flow(im, shuffle=False, batch_size=1))
    p = model.predict(X)
    pred_label = labels[np.argmax(p, axis=1)[0]]
    correct = pred_label == gt_label
    total_time = time.time() - start_time
    return gt_label, pred_label, correct, total_time


def preprocess_whisker_im_to_arr(im: Image):
    im = im.resize((160, 160,), resample=Image.LANCZOS)
    assert im.size == (160, 160,)
    im = rgb2gray(np.array(im))
    im = gaussian(im)
    # im = equalize_adapthist(im)
    thresh = threshold_sauvola(im, window_size=9, k=0.05)
    im = im > thresh
    im = gray2rgb(im)
    im = np.expand_dims(im, 0)
    im = im.astype('uint8')
    im *= 255
    return im


def predict_unprocessed_whisker_url(image_url: str, lion_ids: Iterable[str]) -> dict:
    model, test_datagen, class_indicies, labels = initialize()
    im = download_image(image_url).convert('RGB')
    if im.size != (299, 299,):
        im = im.resize((299, 299,), resample=Image.LANCZOS)
    arr = preprocess_whisker_im_to_arr(im)
    if arr.shape != (1, 299, 299, 3,):
        raise ClassifierError(f'failed processing image for whisker url {image_url} ')
    X = next(test_datagen.flow(arr, shuffle=False, batch_size=1))
    p = model.predict(X)
    predictions = {}
    if lion_ids:
        for i, prob in enumerate(p[0]):
            if labels[i] in lion_ids:
                predictions[labels[i]] = prob
    else:
        for i, prob in enumerate(p[0]):
            predictions[labels[i]] = prob
    return predictions

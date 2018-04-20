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

from linc_cv import *
from linc_cv.ml import download_image, ClassifierError

with open(CLASS_INDICIES_PATH) as f:
    class_indicies = json.load(f)
model = None
labels = [x[0] for x in sorted(class_indicies.items(), key=itemgetter(1))]
num_classes = len(labels)
print(f'num_classes -> {num_classes}')
test_datagen = None


def initialize():
    global model
    global test_datagen
    if model is None:
        model = load_model(WHISKER_FEATURES_PATH)
    if test_datagen is None:
        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            samplewise_center=True,
            samplewise_std_normalization=True, )
    return model, test_datagen


def predict_on_whisker_path(path):
    global model
    global test_datagen
    model, test_datagen = initialize()
    start_time = time.time()
    im = Image.open(path).convert('RGB')
    im = im.resize((299, 299,))
    im = img_to_array(im)
    im = np.expand_dims(im, 0)
    assert im.shape == (1, 299, 299, 3,), im.shape
    gt_label = path.split('/')[-2]
    y = np.zeros((1, num_classes,))
    y[0][class_indicies[gt_label]] = 1
    X = next(test_datagen.flow(im, shuffle=False, batch_size=1))
    p = model.predict(X)
    pred_label = labels[np.argmax(p, axis=1)[0]]
    correct = pred_label == gt_label
    total_time = time.time() - start_time
    return gt_label, pred_label, correct, total_time


def preprocess_whisker_im_to_arr(im):
    im = rgb2gray(np.array(im))
    im = gaussian(im, sigma=2)
    im = im > threshold_sauvola(im, k=0.1)
    im = gray2rgb(im)
    return im


def predict_unprocessed_whisker_url(image_url: str, lion_ids: Iterable[str]) -> dict:
    global model, test_datagen
    model, test_datagen = initialize()

    # lion_id, probability
    im = download_image(image_url).convert('RGB')
    im = im.resize((299, 299,), resample=Image.LANCZOS)
    im = preprocess_whisker_im_to_arr(im)
    im = np.expand_dims(im, 0)
    if im.shape != (1, 299, 299, 3,):
        raise ClassifierError(f'failed processing image for whisker url {image_url} ')
    X = next(test_datagen.flow(im, shuffle=False, batch_size=1))
    p = model.predict(X)
    predictions = {}
    for i, prob in enumerate(p[0]):
        if labels[i] in lion_ids:
            predictions[labels[i]] = prob
    return predictions

import heapq
import time

import numpy as np
from PIL import Image

from linc_cv import INPUT_SHAPE


def predict_on_image_path(*, model, image_path, test_datagen, labels):
    start_time = time.time()
    im = Image.open(image_path).resize(INPUT_SHAPE[:-1], resample=Image.LANCZOS)
    arr = np.array(im)
    arr = np.expand_dims(arr, axis=0)
    assert arr.shape == (1, 299, 299, 3,)
    X = next(test_datagen.flow(arr, shuffle=False, batch_size=1))
    preds = model.predict(X)[0]
    topk = heapq.nlargest(20, range(len(preds)), preds.take)
    topk_labels = [labels[x] for x in topk]
    prediction_time = time.time() - start_time
    return topk_labels, prediction_time


def predict_whisker_url(url):
    return []

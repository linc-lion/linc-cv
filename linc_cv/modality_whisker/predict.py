import numpy as np
from PIL import Image


def predict_on_image_path(*, model, image_path, test_datagen):
    with Image.open(image_path) as im:
        arr = np.array(im)
    X = next(test_datagen.flow(arr, shuffle=False, batch_size=1))
    p = model.predict(X)
    gt_label = None
    topk_labels = None
    prediction_time = None
    return gt_label, topk_labels, prediction_time


def predict_whisker_url(url):
    return []

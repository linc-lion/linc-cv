from io import BytesIO

import numpy as np
import requests
from PIL import Image

from linc_cv import INPUT_SHAPE


def predict_on_url(*, model, image_url, test_datagen, labels, num_results=20):
    r = requests.get(image_url)
    if not r.ok:
        return None
    try:
        im = Image.open(BytesIO(r.content)).convert('RGB')
    except OSError:
        return None
    im = im.resize(INPUT_SHAPE[:-1], resample=Image.LANCZOS)
    arr = np.array(im)
    arr = np.expand_dims(arr, axis=0)
    assert arr.shape == (1, 299, 299, 3,)
    X = next(test_datagen.flow(arr, shuffle=False, batch_size=1))
    return {labels[i]: k for i, k in enumerate(model.predict(X)[0])}


def validate_on_image_path(*, model, image_path, test_datagen, labels):
    im = Image.open(image_path).resize(INPUT_SHAPE[:-1], resample=Image.LANCZOS)
    arr = np.array(im)
    arr = np.expand_dims(arr, axis=0)
    assert arr.shape == (1, 299, 299, 3,)
    X = next(test_datagen.flow(arr, shuffle=False, batch_size=1))
    return {labels[i]: k for i, k in enumerate(model.predict(X)[0])}

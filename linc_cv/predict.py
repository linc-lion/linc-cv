from io import BytesIO

import numpy as np
import requests
from PIL import Image

from linc_cv import INPUT_SHAPE


def predict(image, model, test_datagen, labels, num_results, n_samples):
    image = image.resize(INPUT_SHAPE[:-1], resample=Image.LANCZOS)
    arr = np.array(image)
    arr = np.expand_dims(arr, axis=0)
    assert arr.shape == (1, 299, 299, 3,)
    tgg = test_datagen.flow(arr, shuffle=False, batch_size=n_samples)
    X = np.concatenate([next(tgg) for _ in range(n_samples)])
    preds = model.predict(X)
    preds = np.mean(preds, axis=0)
    topk_pred_idxs = np.argsort(-preds)[:num_results]
    probabilities = preds[topk_pred_idxs].tolist()
    lion_ids = np.array(labels)[topk_pred_idxs].tolist()
    predictions = [{'lion_id': lion_id, 'probability': probability} for lion_id, probability in
                   zip(lion_ids, probabilities)]
    return {'predictions': predictions}


def predict_on_url(*, model, image_url, test_datagen, labels, num_results=20, n_samples=32):
    r = requests.get(image_url)
    if not r.ok:
        return None
    try:
        image = Image.open(BytesIO(r.content)).convert('RGB')
    except OSError:
        return None
    return predict(image, model, test_datagen, labels, num_results, n_samples)


def validate_on_image_path(*, model, image_path, test_datagen, labels, num_results=20, batch_size=32):
    image = Image.open(image_path)
    return predict(image, model, test_datagen, labels, num_results, batch_size)

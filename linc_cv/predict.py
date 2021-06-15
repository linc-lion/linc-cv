from io import BytesIO

import requests
from PIL import Image


def predict(image, clf, num_results):
    # topk_pred_idxs = np.argsort(-preds)[:num_results]
    # probabilities = preds[topk_pred_idxs].tolist()
    # lion_ids = np.array(labels)[topk_pred_idxs].tolist()
    # predictions = [{'lion_id': lion_id, 'probability': probability} for lion_id, probability in
    #                zip(lion_ids, probabilities)]
    # return {'predictions': predictions}
    return {}


def predict_on_url(*, image_url, clf, num_results=20):
    r = requests.get(image_url)
    if not r.ok:
        return None
    try:
        image = Image.open(BytesIO(r.content)).convert('RGB')
    except OSError:
        return None
    return predict(image, clf, num_results)

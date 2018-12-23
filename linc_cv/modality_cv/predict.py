import joblib
from io import BytesIO
import tempfile
from operator import itemgetter

import requests
from PIL import Image

from linc_cv import REDIS_MODEL_RELOAD_KEY, CV_CLASSIFIER_PATH
from .train import CV_NN_Model

cv_model = None
cv_nn_model = None


def predict(image, num_results):
    global cv_model
    global cv_nn_model
    if cv_model is None or cv_nn_model is None:
        cv_model = joblib.load(CV_CLASSIFIER_PATH)
        cv_nn_model = CV_NN_Model()
    feature = cv_nn_model.predict(image)[None, :]
    pl = list(zip(cv_model.classes_, cv_model.predict_proba(feature)[0], ))
    pl = sorted(pl, key=itemgetter(1), reverse=True)[:num_results]
    predictions = [{'lion_id': lion_id, 'probability': probability} for lion_id, probability in pl]
    return {'predictions': predictions}


def predict_cv_url(image_url, num_results=20):
    global cv_model
    global cv_nn_model
    if cv_model is None or cv_nn_model is None:
        cv_model = joblib.load(CV_CLASSIFIER_PATH)
        cv_nn_model = CV_NN_Model()
    r = requests.get(image_url)
    if not r.ok:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg') as ntf:
            image = Image.open(BytesIO(r.content)).convert('RGB')
            image.save(ntf.name, format='JPEG')
            ntf.flush()
            return predict(image=ntf.name, num_results=num_results)
    except OSError:
        return None

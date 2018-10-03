from io import BytesIO

from redis import StrictRedis
import requests
from PIL import Image

from linc_cv import WHISKER_MODEL_PATH_FINAL, WHISKER_CLASSES_LUT_PATH, REDIS_MODEL_RELOAD_KEY
from linc_cv.validation import classifier_classes_lut_to_labels

from .inference import YOLO

whisker_model = None
test_datagen = None
labels = classifier_classes_lut_to_labels(WHISKER_CLASSES_LUT_PATH)


def predict_whisker_url(test_image_url):
    global whisker_model
    global labels
    global test_datagen
    sr = StrictRedis()
    reload_nn_model = sr.get(REDIS_MODEL_RELOAD_KEY)
    if reload_nn_model or whisker_model is None:
        sr.delete(REDIS_MODEL_RELOAD_KEY)
        whisker_model = YOLO(WHISKER_MODEL_PATH_FINAL)

    r = requests.get(test_image_url)
    if not r.ok:
        return None
    buf = BytesIO(r.content)
    try:
        image = Image.open(buf).convert('RGB')
    except OSError:
        return None
    rois = whisker_model.detect_image(image)
    if not rois:
        return None
    print(rois)
    topk_labels = None
    return topk_labels

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from redis import StrictRedis

from linc_cv import WHISKER_MODEL_PATH_FINAL, WHISKER_TESTING_IMAGEDATAGENERATOR_PARAMS, WHISKER_CLASSES_LUT_PATH, REDIS_MODEL_RELOAD_KEY
from linc_cv.predict import predict_on_url
from linc_cv.validation import classifier_classes_lut_to_labels

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
        whisker_model = load_model(WHISKER_MODEL_PATH_FINAL)
    if test_datagen is None:
        test_datagen = ImageDataGenerator(**WHISKER_TESTING_IMAGEDATAGENERATOR_PARAMS)
    topk_labels = predict_on_url(
        model=whisker_model, image_url=test_image_url,
        test_datagen=test_datagen, labels=labels)
    return topk_labels

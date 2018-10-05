from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from redis import StrictRedis

from linc_cv import CV_MODEL_PATH_FINAL, CV_TESTING_IMAGEDATAGENERATOR_PARAMS, CV_CLASSES_LUT_PATH, REDIS_MODEL_RELOAD_KEY
from linc_cv.predict import predict_on_url
from linc_cv.validation import classifier_classes_lut_to_labels

cv_model = None
test_datagen = None
labels = classifier_classes_lut_to_labels(CV_CLASSES_LUT_PATH)


def predict_cv_url(test_image_url):
    global cv_model
    global test_datagen
    sr = StrictRedis()
    reload_nn_model = sr.get(REDIS_MODEL_RELOAD_KEY)
    if reload_nn_model or cv_model is None:
        sr.delete(REDIS_MODEL_RELOAD_KEY)
        cv_model = load_model(CV_MODEL_PATH_FINAL)
    if test_datagen is None:
        test_datagen = ImageDataGenerator(**CV_TESTING_IMAGEDATAGENERATOR_PARAMS)
    topk_labels = predict_on_url(
        model=cv_model, image_url=test_image_url,
        test_datagen=test_datagen, labels=labels)
    return topk_labels

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from linc_cv import CV_MODEL_PATH, CV_TESTING_IMAGEDATAGENERATOR_PARAMS, CV_CLASSES_LUT_PATH
from linc_cv.predict import predict_on_url
from linc_cv.validation import classifier_classes_lut_to_labels

cv_model = None
test_datagen = None
labels = classifier_classes_lut_to_labels(CV_CLASSES_LUT_PATH)


def predict_cv_url(test_image_url):
    global cv_model
    global labels
    global test_datagen
    if cv_model is None:
        cv_model = load_model(CV_MODEL_PATH)
    if test_datagen is None:
        test_datagen = ImageDataGenerator(**CV_TESTING_IMAGEDATAGENERATOR_PARAMS)
    topk_labels = predict_on_url(
        model=cv_model, image_url=test_image_url,
        test_datagen=test_datagen, labels=labels)
    return topk_labels

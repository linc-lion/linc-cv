import json

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from linc_cv import CV_IMAGES_TRAINTEST_PATH, \
    CV_TESTING_IMAGEDATAGENERATOR_PARAMS, CV_MODEL_PATH, \
    CV_CLASSES_LUT_PATH
from linc_cv.validation import classifier_classes_lut_to_labels, validate_classifier


def validate_cv():
    """Validate CV classifier performance on labeled test data"""
    test_datagen = ImageDataGenerator(**CV_TESTING_IMAGEDATAGENERATOR_PARAMS)
    model = load_model(CV_MODEL_PATH)
    labels = classifier_classes_lut_to_labels(CV_CLASSES_LUT_PATH)
    results = validate_classifier(
        traintest_path=CV_IMAGES_TRAINTEST_PATH,
        model=model, test_datagen=test_datagen, labels=labels)
    print(json.dumps(results, indent=4))


def show_processed_cv_activation():
    """show_processed_cv_activation"""
    pass

import json

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from linc_cv import WHISKER_IMAGES_TRAINTEST_PATH, \
    WHISKER_TESTING_IMAGEDATAGENERATOR_PARAMS, WHISKER_MODEL_PATH, \
    WHISKER_CLASSES_LUT_PATH, WHISKER_VALIDATION_JSON_PATH, \
    WHISKER_CLASSIFICATION_REPORT_PATH
from linc_cv.validation import classifier_classes_lut_to_labels, validate_classifier, linc_classification_report


def whisker_test_results():
    test_datagen = ImageDataGenerator(**WHISKER_TESTING_IMAGEDATAGENERATOR_PARAMS)
    model = load_model(WHISKER_MODEL_PATH)
    labels = classifier_classes_lut_to_labels(WHISKER_CLASSES_LUT_PATH)
    results = validate_classifier(
        traintest_path=WHISKER_IMAGES_TRAINTEST_PATH,
        model=model, test_datagen=test_datagen, labels=labels)
    return results


def validate_whisker_classifier():
    """Verify whisker classifier performance on labeled test data.
    Print whisker classifier report and save it to a local file as a pickled Pandas dataframe"""
    results = whisker_test_results()
    with open(WHISKER_VALIDATION_JSON_PATH, 'w') as f:
        json.dump(results, f, indent=4)
    return linc_classification_report(
        results=results, output=WHISKER_CLASSIFICATION_REPORT_PATH)

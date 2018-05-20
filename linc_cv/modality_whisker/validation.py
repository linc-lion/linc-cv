import json

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from linc_cv import WHISKER_IMAGES_TRAINTEST_PATH, \
    WHISKER_TESTING_IMAGEDATAGENERATOR_PARAMS, WHISKER_MODEL_PATH, \
    WHISKER_CLASSES_LUT_PATH, WHISKER_VALIDATION_JSON_PATH
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
    """Verify Whisker classifier performance on labeled test data"""
    results = whisker_test_results()
    with open(WHISKER_VALIDATION_JSON_PATH, 'w') as f:
        json.dump(results, f, indent=4)


def whisker_classifier_report():
    """Print Whisker classifier report and Save it to a local file as a pickled Pandas dataframe"""
    return linc_classification_report(
        results=whisker_test_results(), output='whisker_classification_report.pkl')

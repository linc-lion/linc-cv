import json

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from linc_cv import CV_IMAGES_TRAINTEST_PATH, \
    CV_TESTING_IMAGEDATAGENERATOR_PARAMS, CV_MODEL_PATH, \
    CV_CLASSES_LUT_PATH, WHISKER_VALIDATION_JSON_PATH
from linc_cv.validation import classifier_classes_lut_to_labels, validate_classifier, linc_classification_report


def cv_test_results():
    test_datagen = ImageDataGenerator(**CV_TESTING_IMAGEDATAGENERATOR_PARAMS)
    model = load_model(CV_MODEL_PATH)
    labels = classifier_classes_lut_to_labels(CV_CLASSES_LUT_PATH)
    results = validate_classifier(
        traintest_path=CV_IMAGES_TRAINTEST_PATH,
        model=model, test_datagen=test_datagen, labels=labels)
    return results


def validate_cv_classifier():
    """Verify CV classifier performance on labeled test data"""
    results = cv_test_results()
    with open(WHISKER_VALIDATION_JSON_PATH, 'w') as f:
        json.dump(results, f, indent=4)


def cv_classifier_report():
    """Print CV classifier report and Save it to a local file as a pickled Pandas dataframe"""
    return linc_classification_report(
        results=cv_test_results(), output='cv_classification_report.pkl')

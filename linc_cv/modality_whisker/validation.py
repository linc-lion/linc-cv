import json
from operator import itemgetter

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from linc_cv import WHISKER_IMAGES_TRAINTEST_PATH, \
    WHISKER_TESTING_IMAGEDATAGENERATOR_PARAMS, WHISKER_MODEL_PATH, \
    WHISKER_CLASSES_LUT_PATH
from linc_cv.validation import validate_classifier


def validate_whiskers():
    """Validate whisker classifier performance on labeled test data"""
    test_datagen = ImageDataGenerator(**WHISKER_TESTING_IMAGEDATAGENERATOR_PARAMS)
    model = load_model(WHISKER_MODEL_PATH)
    with open(WHISKER_CLASSES_LUT_PATH) as f:
        class_indicies = json.load(f)
    labels = [x[0] for x in sorted(class_indicies.items(), key=itemgetter(1))]
    results = validate_classifier(
        traintest_path=WHISKER_IMAGES_TRAINTEST_PATH,
        model=model, test_datagen=test_datagen, labels=labels)
    print(json.dumps(results, indent=4))


def show_processed_whisker_activation():
    """Validate whiskers on holdout test data."""
    pass

import json

from linc_cv import WHISKER_IMAGES_TRAINTEST_PATH
from linc_cv.validation import validate_classifier_on_testdir


def validate_whiskers():
    """Validate whisker classifier performance on labeled test data"""
    results = validate_classifier_on_testdir(traintest_path=WHISKER_IMAGES_TRAINTEST_PATH)
    print(json.dumps(results))


def show_processed_whisker_activation():
    """Validate whiskers on holdout test data."""
    pass

import os
import os.path
import json

from linc_cv.modality_cv.predict import predict
from linc_cv import CV_IMAGES_TRAINTEST_PATH, CV_VALIDATION_JSON_PATH


def cv_test_results():
    results = []
    for root, dirs, files in os.walk(os.path.join(CV_IMAGES_TRAINTEST_PATH, 'test')):
        for f in files:
            image_path = os.path.join(root, f)
            gt_label = image_path.split(os.path.sep)[-2]
            p = predict(image, num_results)
            results.append([gt_label, p])
    return results


def validate_cv_classifier():
    """Verify CV classifier performance on labeled test data.
    Print CV classifier report and save it to a local file as a pickled Pandas dataframe"""
    results = cv_test_results()
    with open(CV_VALIDATION_JSON_PATH, 'w') as f:
        json.dump(results, f, indent=4)

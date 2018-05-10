# coding=utf-8
import json
import os
import sys

from linc_cv import WHISKER_IMAGES_TRAINTEST_PATH, WHISKER_IMAGES_PATH
from linc_cv.whiskers.predict import predict_whisker_from_preprocessed_image


def validate_whiskers(all_whiskers=False):
    """
    Validate whisker classifier performance on labeled test data
    """
    results = []
    prediction_times = []
    if all_whiskers:
        rootdir = WHISKER_IMAGES_PATH
    else:
        rootdir = WHISKER_IMAGES_TRAINTEST_PATH
    for root, dirs, files in os.walk(rootdir):
        for f in files:
            path = os.path.join(root, f)
            gt_label, topk_labels, prediction_time = predict_whisker_from_preprocessed_image(path)
            results.append([gt_label, topk_labels])
            prediction_times.append(prediction_time)
            sys.stdout.write('.')
            sys.stdout.flush()
    print(json.dumps(results))

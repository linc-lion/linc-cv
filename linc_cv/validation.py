import os
import sys

from linc_cv.whiskers.predict import predict_whisker_path


def validate_classifier_on_testdir(*, traintest_path):
    results = []
    prediction_times = []
    for root, dirs, files in os.walk(os.path.join(traintest_path, 'test')):
        for f in files:
            gt_label, topk_labels, prediction_time = predict_whisker_path(os.path.join(root, f))
            results.append([gt_label, topk_labels])
            prediction_times.append(prediction_time)
            sys.stdout.write('.')
            sys.stdout.flush()
    return results
# coding=utf-8
import os
from collections import defaultdict

import numpy as np

from linc_cv import WHISKER_IMAGES_TRAINTEST_PATH
from linc_cv.whiskers.predict import predict_whisker_from_preprocessed_image


def validate_test_whiskers():
    classifications = defaultdict(list)
    all_times = []
    all_scores = []
    for root, dirs, files in os.walk(WHISKER_IMAGES_TRAINTEST_PATH):
        for f in files:
            path = os.path.join(root, f)
            gt_label, pred_label, correct, total_time = predict_whisker_from_preprocessed_image(path)
            all_scores.append(correct)
            all_times.append(total_time)
            classifications[gt_label].append(pred_label)

    for gt_label, pred_labels in classifications.items():
        scores = []
        for pred_label in pred_labels:
            scores.append(pred_label == gt_label)
        print(f'label: {gt_label}, accuracy: {np.around(np.mean(scores), 3)}')
    print(f'OVERALL, number of test samples: {len(all_scores)}, '
          f'accuracy: {np.around(np.mean(all_scores), 3)}')
    print(f'OVERALL, mean time to perform one '
          f'prediction: {np.around(np.mean(all_times), 3)}')

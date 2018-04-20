# coding=utf-8

import random
from collections import defaultdict

import numpy as np

from linc_cv.ml import initialize, ClassifierError, predict_lion


def new_test_lion(lion_id_count=5):
    global linc_features, features_lut, model
    linc_features, features_lut, model = initialize()
    feature_types = list(features_lut.keys())
    while True:
        feature_type = random.choice(feature_types)
        all_lion_ids = list(features_lut[feature_type].keys())
        random.shuffle(all_lion_ids)
        selected_lion_ids = [all_lion_ids.pop() for _ in range(lion_id_count)]
        gt_class = random.choice(selected_lion_ids)
        gt_class_idxs = features_lut[feature_type][gt_class]
        if len(gt_class_idxs) < 3:
            continue  # skip lion ids with too few samples to make a test case
        test_feature_idx = random.choice(gt_class_idxs)
        return gt_class, test_feature_idx, feature_type, selected_lion_ids


def validate_random_lion(*args):
    try:
        gt_class, test_feature_idx, feature_type, lion_ids = new_test_lion()
        return predict_lion(feature_type, lion_ids, gt_class=gt_class, test_feature_idx=test_feature_idx)
    except ClassifierError as e:
        print(e.message)


def validate_random_lions():
    """
    Continuously pick a lion at random, pick a feature at random, then
    generate a holdout test set for that particular lion and measure
    the performance of the classifier for that lion
    """

    scores = defaultdict(list)
    val_accs = defaultdict(list)
    while True:
        try:
            feature_type, correct, val_acc, probas, labels = validate_random_lion()
        except TypeError:
            continue
        scores[feature_type].append(correct)
        val_accs[feature_type].append(val_acc)
        print('~' * 100)
        for feature_type in scores:
            val_mean = np.round(np.mean(val_accs[feature_type]), 3)
            num_tests = len(scores[feature_type])
            score = np.round(np.mean(scores[feature_type]), 3)
            print(f'feature {feature_type}: test accuracy mean -> {score}, '
                  f'number of tests -> {num_tests}, '
                  f'validation set holdout accuracy mean -> {val_mean}')
        print('~' * 120)

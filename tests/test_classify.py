# coding=utf-8

import multiprocessing
import random
from collections import defaultdict

import numpy as np

from linc_cv.classify import initialize, ClassifierError, test_lion


def get_test_lion(lion_id_count=5):
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


class InfiniteList():
    def __iter__(self):
        return self

    def __next__(self):
        return None


def test_random_lion(*args):
    try:
        gt_class, test_feature_idx, feature_type, lion_ids = get_test_lion()
        return test_lion(feature_type, lion_ids, gt_class=gt_class, test_feature_idx=test_feature_idx)
    except ClassifierError as e:
        print(e.message)


def test_random_lions_mp():
    scores = defaultdict(list)
    val_accs = defaultdict(list)
    infinitelist = InfiniteList()
    process_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=process_count) as pool:
        for result in pool.imap_unordered(test_random_lion, infinitelist):
            try:
                feature_type, correct, val_acc, probas, labels = result
            except TypeError:
                continue
            scores[feature_type].append(correct)
            val_accs[feature_type].append(val_acc)
            print('~' * 100)
            for feature_type in scores:
                score = np.round(np.mean(scores[feature_type]), 3)
                print(f'feature {feature_type}: score -> {score}, '
                      f'score std -> {np.round(np.std(scores[feature_type]), 3)}, '
                      f'n_scores -> {len(scores[feature_type])}, '
                      f'val_acc mean -> {np.round(np.mean(val_accs[feature_type]), 3)}, '
                      f'val_acc std -> {np.round(np.std(val_accs[feature_type]), 3)}')
            print('~' * 100)


def test_random_lions():
    scores = defaultdict(list)
    val_accs = defaultdict(list)
    while True:
        try:
            feature_type, correct, val_acc, probas, labels = test_random_lion()
        except TypeError:
            continue
        scores[feature_type].append(correct)
        val_accs[feature_type].append(val_acc)
        print('~' * 100)
        for feature_type in scores:
            score = np.round(np.mean(scores[feature_type]), 3)
            print(f'feature {feature_type}: score -> {score}, '
                  f'score std -> {np.round(np.std(scores[feature_type]), 3)}, '
                  f'n_scores -> {len(scores[feature_type])}, '
                  f'val_acc mean -> {np.round(np.mean(val_accs[feature_type]), 3)}, '
                  f'val_acc std -> {np.round(np.std(val_accs[feature_type]), 3)}')
        print('~' * 100)


if __name__ == '__main__':
    test_image_url = 'http://pixdaus.com/files/items/pics/7/84/542784_81cef138c75698faddafee92d42c0cc5_large.jpg'
    feature_type = 'main-id'
    lion_ids = ['131', '234', '142', '97', '163']
    try:
        feature_type, correct, val_acc, probas, labels = test_lion(
            feature_type=feature_type, lion_ids=lion_ids, test_image_url=test_image_url)
    except ClassifierError as e:
        print(e.message)

    # test_random_lion()
    # test_random_lions()
    # test_random_lions_mp()

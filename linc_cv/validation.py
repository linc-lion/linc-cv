import json
import os
from operator import itemgetter

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

from linc_cv.predict import validate_on_image_path


def classifier_classes_lut_to_labels(lut_path):
    try:
        with open(lut_path) as f:
            class_indicies = json.load(f)
    except FileNotFoundError:
        return None
    labels = [x[0] for x in sorted(class_indicies.items(), key=itemgetter(1))]
    return labels


def validate_classifier(*, traintest_path, model, test_datagen, labels):
    results = []
    prediction_times = []
    for root, dirs, files in os.walk(os.path.join(traintest_path, 'test')):
        for f in files:
            image_path = os.path.join(root, f)
            gt_label = image_path.split(os.path.sep)[-2]
            topk_labels, prediction_time = validate_on_image_path(
                model=model, image_path=image_path, test_datagen=test_datagen,
                labels=labels)
            results.append([gt_label, topk_labels])
            prediction_times.append(prediction_time)
    return results


def linc_classification_report(*, results, output):
    """CV classification report"""
    y_true, y_pred = zip(*([x, y[0]] for x, y in results))
    print(classification_report(y_true, y_pred))
    prfs_labels = sorted(list(set(y_true + y_pred)))
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_pred)
    df = pd.DataFrame(
        {'label': prfs_labels, 'precision': precision,
         'recall': recall, 'fbeta_score': fbeta_score,
         'support': support})
    df = df.set_index('label')
    df.to_pickle(output)
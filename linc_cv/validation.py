import os
import sys

from linc_cv.modality_whisker.predict import predict_on_image_path


def validate_classifier(*, traintest_path, model, test_datagen):
    results = []
    prediction_times = []
    for root, dirs, files in os.walk(os.path.join(traintest_path, 'test')):
        for f in files:
            image_path = os.path.join(root, f)
            gt_label, topk_labels, prediction_time = predict_on_image_path(
                model=model, image_path=image_path, test_datagen=test_datagen)
            results.append([gt_label, topk_labels])
            prediction_times.append(prediction_time)
            sys.stdout.write('.')
            sys.stdout.flush()
    return results

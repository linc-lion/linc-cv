from celery import Celery

from linc_cv import ClassifierError, WHISKER_CLASSIFIER_ACCURACY
from linc_cv.whiskers.predict import predict_whisker_url

c = Celery()
c.conf.broker_url = 'redis://localhost:6379/0'
c.conf.result_backend = 'redis://localhost:6379/0'


@c.task()
def classify_image_url(test_image_url, feature_type):
    # FIXME
    results = []
    try:
        if 'whisker' in feature_type:
            predictions = predict_whisker_url(test_image_url)
            for lion_id, probability in predictions.items():
                results.append(
                    {'classifier': round(float(WHISKER_CLASSIFIER_ACCURACY), 3),
                     'confidence': round(float(probability), 3),
                     'id': int(lion_id)})
        else:
            feature_type, correct, val_acc, probas, labels = predict_lion(
                feature_type=feature_type, test_image_url=test_image_url)

            for lion_id, probability in zip(labels, probas[0]):
                results.append(
                    {'classifier': round(float(val_acc), 3),
                     'confidence': round(float(probability), 3),
                     'id': int(lion_id)})

    except ClassifierError as e:
        return {
            'status': 'error',
            'info': e.message}

    return {
        'status': 'finished',
        'match_probability': results}

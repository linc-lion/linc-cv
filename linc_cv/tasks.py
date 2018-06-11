from celery import Celery

from linc_cv import ClassifierError
from linc_cv.modality_cv.predict import predict_cv_url
from linc_cv.modality_whisker.predict import predict_whisker_url

c = Celery()
c.conf.broker_url = 'redis://localhost:6379/0'
c.conf.result_backend = 'redis://localhost:6379/0'


@c.task(acks_late=True)
def classify_image_url(test_image_url, feature_type):
    try:
        if 'whisker' in feature_type:
            results = predict_whisker_url(test_image_url)
        elif feature_type == 'cv':
            results = predict_cv_url(test_image_url)
        else:
            raise ClassifierError(f'unknown feature_type {feature_type}')
        return {
            'status': 'finished',
            'predictions': results}
    except ClassifierError as e:
        return {
            'status': 'error',
            'info': e.message}

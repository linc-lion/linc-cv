import requests
from celery import Celery

from linc_cv.modality_cv.download import download_cv_images
from linc_cv.modality_cv.predict import predict_cv_url
from linc_cv.modality_cv.train import extract_cv_features
from linc_cv.modality_whisker.predict import predict_whisker_url
from linc_cv.parse_lion_db import parse_lion_database
from linc_cv.settings import ClassifierError

c = Celery(backend='redis://localhost:6379/0', broker='redis://localhost:6379/0')
c.conf.task_track_started = True
c.conf.task_time_limit = 60 * 10  # 10 minutes expiration
c.conf.result_expires = 60 * 10  # 10 minutes expiration
c.conf.task_routes = {
    'linc_cv.tasks.retrain': {'queue': 'training'},
    'linc_cv.tasks.classify_image_url': {'queue': 'classification'}}


@c.task(track_started=True, acks_late=True)
def retrain():
    print('parsing lion database')
    parse_lion_database()

    print('downloading cv images')
    download_cv_images(mp=False)
    print('training cv classifier')
    extract_cv_features()

    # print('downloading whisker images')
    # download_whisker_images(mp=False)

    # TODO: recompute whisker spot lookup table on demand
    # print('training whisker classifier')
    # train_whisker_classifier(mp=False)


@c.task(track_started=True, acks_late=True)
def classify_image_url(test_image_url, feature_type):
    try:
        if 'whisker' in feature_type:
            results = predict_whisker_url(test_image_url)
        elif feature_type == 'cv':
            results = predict_cv_url(test_image_url)
        else:
            raise ClassifierError(f'unknown feature_type {feature_type}')
        try:
            return {
                'status': 'finished', **results}
        except TypeError as te:
            print("General Classification error: {0}".format(te))
            raise ClassifierError('General classification failure. Contact support.')
    except ClassifierError as e:
        return {
            'status': 'error',
            'info': e.message}
    except requests.exceptions.SSLError:
        return {
            'status': 'error',
            'info': 'Failed to download image from lion image host due to an SSL error'}

from celery import Celery

from linc_cv import ClassifierError
from linc_cv.modality_cv.download import download_cv_images
from linc_cv.modality_cv.predict import predict_cv_url
from linc_cv.modality_cv.train import train_cv_classifier
from linc_cv.modality_whisker.download import download_whisker_images
from linc_cv.modality_whisker.predict import predict_whisker_url
from linc_cv.modality_whisker.train import train_whisker_classifier
from linc_cv.parse_lion_db import parse_lion_database

c = Celery(backend='redis://localhost:6379/0', broker='redis://localhost:6379/0')
c.conf.task_track_started = True


@c.task(track_started=True)
def retrain():
    # TODO: download database and save it here
    print('parsing lion database')
    parse_lion_database('/home/adam/lion-db-dump-2018-07-11T01-21-10.json')

    print('downloading cv images')
    download_cv_images(mp=False)
    print('training cv classifier')
    train_cv_classifier(mp=False)

    print('downloading whisker images')
    download_whisker_images(mp=False)
    print('training whisker classifier')
    train_whisker_classifier(mp=False)


@c.task(track_started=True)
def classify_image_url(test_image_url, feature_type):
    try:
        if 'whisker' in feature_type:
            results = predict_whisker_url(test_image_url)
        elif feature_type == 'cv':
            results = predict_cv_url(test_image_url)
        else:
            raise ClassifierError(f'unknown feature_type {feature_type}')
        return {
            'status': 'finished', **results}
    except ClassifierError as e:
        return {
            'status': 'error',
            'info': e.message}

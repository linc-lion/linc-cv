import argparse
import inspect
import os
import sys
from subprocess import run

from linc_cv import BASE_DIR
from linc_cv.parse_lion_db import parse_lion_database
from linc_cv.web import app
from linc_cv.modality_whisker.download import download_whisker_images
from linc_cv.modality_whisker.train import train_whisker_classifier
from linc_cv.modality_whisker.validation import validate_whisker_classifier
from linc_cv.modality_cv.download import download_cv_images
from linc_cv.modality_cv.train import train_cv_classifier
from linc_cv.modality_cv.validation import validate_cv_classifier

CELERY_EXE_PATH = os.path.join(os.path.dirname(sys.argv[0]), 'celery')
FLOWER_EXE_PATH = os.path.join(os.path.dirname(sys.argv[0]), 'flower')


def main():
    """
    linc_cv: command line interface entry point
    """
    parser = argparse.ArgumentParser(
        description='LINC Lion Recognition System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--parse-lion-database',
        help=inspect.getdoc(parse_lion_database))

    # < feature cv specific >
    parser.add_argument(
        '--download-cv-images', action='store_true',
        help=inspect.getdoc(download_cv_images))
    parser.add_argument(
        '--train-cv-classifier', action='store_true',
        help=inspect.getdoc(train_cv_classifier))
    parser.add_argument(
        '--validate-cv-classifier', action='store_true',
        help=inspect.getdoc(validate_cv_classifier))

    # </ feature cv specific >

    # < whisker specific >

    parser.add_argument(
        '--download-whisker-images', action='store_true',
        help=inspect.getdoc(download_whisker_images))
    parser.add_argument(
        '--train-whisker-classifier', action='store_true',
        help=inspect.getdoc(train_whisker_classifier))
    parser.add_argument(
        '--validate-whisker-classifier', action='store_true',
        help=inspect.getdoc(validate_whisker_classifier))

    # </ whisker specific >

    parser.add_argument(
        '--web', action='store_true',
        help="Start HTTP REST API")
    parser.add_argument(
        '--worker-training', action='store_true',
        help="Training queue worker")
    parser.add_argument(
        '--worker-classification', action='store_true',
        help="Classification queue worker")
    parser.add_argument(
        '--flower', action='store_true',
        help="Start API task worker monitor (Celery Flower)")

    args = parser.parse_args()

    if args.parse_lion_database:
        parse_lion_database(db_json_path=args.parse_lion_database)

    # < feature cv specific >

    if args.download_cv_images:
        download_cv_images()

    if args.train_cv_classifier:
        train_cv_classifier()

    if args.validate_cv_classifier:
        validate_cv_classifier()

    # </ feature cv specific >

    # < whisker specific >

    if args.download_whisker_images:
        download_whisker_images()

    if args.train_whisker_classifier:
        train_whisker_classifier()

    if args.validate_whisker_classifier:
        validate_whisker_classifier()

    # </ whisker specific >

    if args.web:
        app.run(host='0.0.0.0', port=5000, debug=False)

    if args.worker_training:
        cmd = f'{CELERY_EXE_PATH} worker -A linc_cv.tasks --concurrency 1 ' \
              f'-Q training --max-tasks-per-child=512 -E -n training@%h'.split(' ')
        run(cmd, check=True, cwd=BASE_DIR)

    if args.worker_classification:
        cmd = f'{CELERY_EXE_PATH} worker -A linc_cv.tasks --concurrency 1 ' \
              f'-Q classification --max-tasks-per-child=512 -E -n classification@%h'.split(' ')
        run(cmd, check=True, cwd=BASE_DIR)

    if args.flower:
        cmd = f'{FLOWER_EXE_PATH} flower -A linc_cv.tasks --address=0.0.0.0 --port=5555'.split(' ')
        run(cmd, check=True, cwd=BASE_DIR)

import argparse
import inspect
import os
import sys
from subprocess import run

import matplotlib

matplotlib.use('Agg')
from linc_cv import BASE_DIR
from linc_cv.parse_lion_db import generate_images_lut
from linc_cv.scrape_lion_db import scrape_lion_database
from linc_cv.web import app
from linc_cv.modality_whisker.download import download_whisker_images
from linc_cv.modality_whisker.train import train_whisker_classifier
from linc_cv.modality_whisker.validation import validate_whisker_classifier, whisker_classifier_report
from linc_cv.modality_cv.download import download_cv_images
from linc_cv.modality_cv.train import train_cv_classifier
from linc_cv.modality_cv.validation import validate_cv_classifier, cv_classifier_report

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
        '--scrape-lion-database', action='store_true',
        help=inspect.getdoc(scrape_lion_database))
    parser.add_argument(
        '--generate-images-lut', action='store_true',
        help=inspect.getdoc(generate_images_lut))
    parser.add_argument(
        '--scraping-max-id', type=int, default=3000,
        help='LINC database scrape id limit')

    # < feature cv specific >
    parser.add_argument(
        '--download-cv-images', action='store_true',
        help=inspect.getdoc(download_cv_images))
    parser.add_argument(
        '--train-cv-classifier', action='store_true',
        help=inspect.getdoc(train_cv_classifier))
    parser.add_argument(
        '--cv-classifier-report', action='store_true',
        help=inspect.getdoc(cv_classifier_report))
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
        '--whisker-classifier-report', action='store_true',
        help=inspect.getdoc(whisker_classifier_report))
    parser.add_argument(
        '--validate-whisker-classifier', action='store_true',
        help=inspect.getdoc(validate_whisker_classifier))

    # </ whisker specific >

    parser.add_argument(
        '--web', action='store_true',
        help="Start HTTP REST API")
    parser.add_argument(
        '--worker', action='store_true',
        help="Start API task worker (at least one must always "
             "be present for HTTP REST API to function properly.")
    parser.add_argument(
        '--flower', action='store_true',
        help="Start API task worker monitor (Celery Flower)")

    args = parser.parse_args()
    if args.scrape_lion_database:
        scrape_lion_database(
            max_lion_id=args.max_lion_id)

    if args.generate_images_lut:
        generate_images_lut()

    # < feature cv specific >

    if args.download_cv_images:
        download_cv_images()

    if args.train_cv_classifier:
        train_cv_classifier()

    if args.cv_classifier_report:
        cv_classifier_report()

    if args.validate_cv_classifier:
        validate_cv_classifier()

    # </ feature cv specific >

    # < whisker specific >

    if args.download_whisker_images:
        download_whisker_images()

    if args.train_whisker_classifier:
        train_whisker_classifier()

    if args.whisker_classifier_report:
        whisker_classifier_report()

    if args.validate_whisker_classifier:
        validate_whisker_classifier()

    # </ whisker specific >

    if args.web:
        app.run(host='0.0.0.0', port=5000, debug=False)

    if args.worker:
        cmd = f'{CELERY_EXE_PATH} worker -A linc_cv.tasks --concurrency=1 --max-tasks-per-child=64 -E'.split(' ')
        run(cmd, check=True, cwd=BASE_DIR)

    if args.flower:
        cmd = f'{FLOWER_EXE_PATH} flower -A linc_cv.tasks --address=0.0.0.0 --port=5555'.split(' ')
        run(cmd, check=True, cwd=BASE_DIR)

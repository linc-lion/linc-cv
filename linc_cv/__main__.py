# coding=utf-8
import argparse
import inspect
import os
import sys
from subprocess import run

import matplotlib

matplotlib.use('Agg')
from linc_cv import BASE_DIR
from linc_cv.parse_lion_db import linc_db_to_image_lut
from linc_cv.scrape_lion_db import scrape_lion_database
from linc_cv.web import app
from linc_cv.whiskers.download import download_whisker_images
from linc_cv.whiskers.process import process_whisker_images, show_random_processed_whisker_image, \
    show_random_processed_whisker_activations
from linc_cv.whiskers.train import train_whiskers
from linc_cv.whiskers.train_test_split import whiskers_train_test_split
from linc_cv.whiskers.validation import validate_whiskers
from linc_cv.feature_cv.download import download_cv_images
from linc_cv.feature_cv.train import train_cv

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
        '--max-lion-id', type=int, default=2000,
        help='Maximum lion id in LINC database')
    parser.add_argument(
        '--parse-lion-database', action='store_true',
        help=inspect.getdoc(linc_db_to_image_lut))

    # < feature cv specific >
    parser.add_argument(
        '--download-cv-images', action='store_true',
        help=inspect.getdoc(download_cv_images))
    parser.add_argument(
        '--train-cv', action='store_true',
        help=inspect.getdoc(train_cv))
    # parser.add_argument(
    #     '--validate-cv-test-set', action='store_true',
    #     help="Validate 'cv' images on holdout test data.")
    # parser.add_argument(
    #     '--validate-cv-all', action='store_true',
    #     help="Validate 'cv' images on entire dataset, including training data.")
    # parser.add_argument(
    #     '--show-random-processed-cv-activations', action='store_true',
    #     help=inspect.getdoc(show_random_processed_cv_activations))

    # </ feature cv specific >

    # < whisker specific >

    parser.add_argument(
        '--download-whisker-images', action='store_true',
        help=inspect.getdoc(download_whisker_images))
    parser.add_argument(
        '--show-random-processed-whisker-image', action='store_true',
        help=inspect.getdoc(show_random_processed_whisker_image))
    parser.add_argument(
        '--process-whisker-images', action='store_true',
        help=inspect.getdoc(process_whisker_images))
    parser.add_argument(
        '--whiskers-train-test-split', action='store_true',
        help=inspect.getdoc(whiskers_train_test_split))
    parser.add_argument(
        '--train-whiskers', action='store_true',
        help=inspect.getdoc(train_whiskers))
    parser.add_argument(
        '--validate-whiskers-test-set', action='store_true',
        help="Validate whiskers on holdout test data.")
    parser.add_argument(
        '--validate-whiskers-all', action='store_true',
        help="Validate whiskers on entire dataset, including training data.")
    parser.add_argument(
        '--show-random-processed-whisker-activations', action='store_true',
        help=inspect.getdoc(show_random_processed_whisker_activations))

    # </ whisker specific >

    parser.add_argument(
        '--no-validation', action='store_false',
        help="Do not perform cross-validation. Useful for final training.")
    parser.add_argument(
        '--epochs', type=int, default=500,
        help="Upper bound on training epochs.")
    parser.add_argument(
        '--class-weight-smoothing-factor', type=float, default=0.1,
        help="Factor that smooths extremely uneven weights computed for "
             "balanced class weights.")
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

    if args.parse_lion_database:
        linc_db_to_image_lut()

    # if args.extract_lion_features:
    #     generate_linc_lut()
    # if args.validate_random_lions:
    #     validate_random_lions()

    # < feature cv specific >

    if args.download_cv_images:
        download_cv_images()

    if args.train_cv:
        train_cv()

    # </ feature cv specific >

    # < whisker specific >

    if args.download_whisker_images:
        download_whisker_images()

    if args.show_random_processed_whisker_activations:
        show_random_processed_whisker_activations()

    if args.show_random_processed_whisker_image:
        show_random_processed_whisker_image()

    if args.process_whisker_images:
        process_whisker_images()

    if args.whiskers_train_test_split:
        whiskers_train_test_split()

    if args.train_whiskers:
        train_whiskers(args.no_validation, args.epochs, args.class_weight_smoothing_factor)

    if args.validate_whiskers_test_set:
        validate_whiskers(all_whiskers=False)

    if args.validate_whiskers_all:
        validate_whiskers(all_whiskers=True)

    # </ whisker specific >

    if args.web:
        app.run(host='0.0.0.0', port=5000, debug=False)

    if args.worker:
        cmd = f'{CELERY_EXE_PATH} worker -A linc_cv.tasks --concurrency=1 --max-tasks-per-child=4 -E'.split(' ')
        run(cmd, check=True, cwd=BASE_DIR)

    if args.flower:
        cmd = f'{FLOWER_EXE_PATH} flower -A linc_cv.tasks --address=0.0.0.0 --port=5555'.split(' ')
        run(cmd, check=True, cwd=BASE_DIR)

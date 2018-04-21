# coding=utf-8
import argparse
import inspect

from linc_cv.ml import generate_linc_lut
from linc_cv.parse_lion_db import linc_db_to_image_lut
from linc_cv.scrape_lion_db import scrape_lion_database
from linc_cv.validation_ml import validate_random_lions
from linc_cv.whiskers.download import download_whisker_images
from linc_cv.whiskers.process import process_whisker_images, show_random_processed_whisker_image
from linc_cv.whiskers.train import train_whiskers
from linc_cv.whiskers.train_test_split import whiskers_train_test_split


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
    parser.add_argument(
        '--extract-lion-features', action='store_true',
        help=inspect.getdoc(generate_linc_lut))
    parser.add_argument(
        '--validate-random-lions', action='store_true',
        help=inspect.getdoc(validate_random_lions))
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
        '--train_whiskers', action='store_true',
        help=inspect.getdoc(train_whiskers))

    args = parser.parse_args()
    if args.scrape_lion_database:
        scrape_lion_database(
            max_lion_id=args.max_lion_id)

    if args.parse_lion_database:
        linc_db_to_image_lut()

    if args.extract_lion_features:
        generate_linc_lut()

    if args.validate_random_lions:
        validate_random_lions()

    if args.download_whisker_images:
        download_whisker_images()

    if args.show_random_processed_whisker_image:
        show_random_processed_whisker_image()

    if args.process_whisker_images:
        process_whisker_images()

    if args.whiskers_train_test_split:
        whiskers_train_test_split()

    if args.train_whiskers:
        train_whiskers()

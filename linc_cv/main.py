import argparse

from linc_cv.modality_cv.download import download_cv_images
from linc_cv.modality_cv.train import extract_cv_features, train_cv_classifier
from linc_cv.modality_whisker.download import download_whisker_images
from linc_cv.modality_whisker.train import train_whisker_classifier
from linc_cv.parse_lion_db import parse_lion_database


def main():
    """
    linc_cv: command line interface entry point
    """
    parser = argparse.ArgumentParser(
        description='LINC Lion Recognition System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--parse-lion-database', action='store_true')

    # < feature cv specific >
    parser.add_argument(
        '--download-cv-images', action='store_true')
    parser.add_argument(
        '--extract-cv-features', action='store_true')
    parser.add_argument(
        '--train-cv-classifier', action='store_true')

    # </ feature cv specific >

    # < whisker specific >

    parser.add_argument(
        '--download-whisker-images', action='store_true', )
    parser.add_argument(
        '--train-whisker-classifier', action='store_true', )

    # </ whisker specific >

    args = parser.parse_args()

    if args.parse_lion_database:
        parse_lion_database()

    # < feature cv specific >

    if args.download_cv_images:
        download_cv_images()

    if args.extract_cv_features:
        extract_cv_features()

    if args.train_cv_classifier:
        train_cv_classifier()

    # </ feature cv specific >

    # < whisker specific >

    if args.download_whisker_images:
        download_whisker_images()

    if args.train_whisker_classifier:
        train_whisker_classifier()

    # </ whisker specific >


if __name__ == '__main__':
    main()

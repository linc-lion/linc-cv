# coding=utf-8
import argparse
import inspect
import json
import multiprocessing
import os

from linc_cv.parse_lion_db import linc_db_to_image_lut
from linc_cv.scrape_lion_db import scrape_lion_idx
from . import BASE_DIR


def scrape_lion_database(max_lion_id):
    """
    Scrape LINC database containing lion_ids and associated image
    urls for all features using brute force and save the resulting
    entries into a JSON file for parsing.

    The database must be downloaded for any training task to succeed.

    :param max_lion_id: the largest id of any lion in the database
    """

    data = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for result in pool.imap_unordered(scrape_lion_idx, list(range(max_lion_id))):
            data.append(result)

    with open(os.path.join(BASE_DIR, 'data', 'linc_db.json'), 'w') as f:
        json.dump(data, f)

    print('LINC database scraping succeeded.')


def main():
    """
    linc_cv command line interface entry point
    """
    parser = argparse.ArgumentParser(
        description='LINC Computer Vision System',
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

    args = parser.parse_args()
    if args.scrape_lion_database:
        scrape_lion_database(
            max_lion_id=args.max_lion_id)

    if args.parse_lion_database:
        linc_db_to_image_lut()
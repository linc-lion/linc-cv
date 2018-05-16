import json
import multiprocessing
import sys

import requests

from linc_cv import LINC_DB_PATH


def scrape_lion_idx(idx):
    j = None
    try:
        j = requests.get('https://linc-api.herokuapp.com/lions/' + str(idx)).json()
        sys.stdout.write('.')
    except:
        sys.stdout.write('-')
    sys.stdout.flush()
    return j


def scrape_lion_database(max_lion_id):
    """
    Scrape LINC database containing lion_ids and associated image
    urls for all features using brute force and save the resulting
    entries into a JSON file for parsing.

    The database must be downloaded for any training task to succeed.

    :param max_lion_id: the largest id of any lion in the database
    """

    data = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 4) as pool:
        for result in pool.imap_unordered(scrape_lion_idx, list(range(max_lion_id))):
            data.append(result)

    with open(LINC_DB_PATH, 'w') as f:
        json.dump(data, f)

    print('LINC database scraping succeeded.')

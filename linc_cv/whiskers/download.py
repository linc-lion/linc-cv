# coding=utf-8

import json
import multiprocessing
import os
import shutil
import sys

import requests

from linc_cv import datapath, IMAGES_LUT_PATH


def download_whisker_image(image_url, lion_id, idx):
    filepath = datapath(['whisker_images', f'{lion_id}/{idx}'])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        r = requests.get(image_url, stream=True)
        shutil.copyfileobj(r.raw, f)
    sys.stdout.write('.')
    sys.stdout.flush()


def download_whisker_images():
    """
    Download all whisker images for processing and training a
    new whisker classifier
    """

    with open(IMAGES_LUT_PATH) as f:
        images_lut = json.load(f)

    data = []
    i = 0
    for lion_id in images_lut:
        try:
            for url in images_lut[lion_id]['whisker']:
                data.append((url, lion_id, i,))
                i += 1
        except KeyError:
            continue

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2) as pool:
        pool.starmap(download_whisker_image, data)

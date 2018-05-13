# coding=utf-8

import json
import multiprocessing
import os
import shutil
import sys
from io import BytesIO

import requests
from PIL import Image

from linc_cv import WHISKER_IMAGES_PATH, IMAGES_LUT_PATH


def download_linc_image(image_url, lion_id, idx):
    filepath = os.path.join(WHISKER_IMAGES_PATH, f'{lion_id}/{idx}.jpg')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    r = requests.get(image_url)
    if r.ok:
        try:
            Image.open(BytesIO(r.content)).convert('RGB').save(
                filepath, format='JPEG', optimize=True)
        except OSError:
            pass
        else:
            sys.stdout.write('.')
            sys.stdout.flush()


def download_whisker_images():
    """
    Download all whisker images for processing and training a
    new whisker classifier
    """

    try:
        shutil.rmtree(WHISKER_IMAGES_PATH)
    except FileNotFoundError:
        pass

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
        pool.starmap(download_linc_image, data)

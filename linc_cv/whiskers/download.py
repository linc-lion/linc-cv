# coding=utf-8

import json
import multiprocessing
import os
import shutil
import sys

import requests


def download_whisker_image(image_url, lion_id, idx):
    filepath = f'data/whisker_images/{lion_id}/{idx}'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        r = requests.get(image_url, stream=True)
        shutil.copyfileobj(r.raw, f)
    sys.stdout.write('.')
    sys.stdout.flush()


if __name__ == '__main__':
    with open('data/images_lut.json') as f:
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

    with multiprocessing.Pool(processes=32) as pool:
        pool.starmap(download_whisker_image, data)

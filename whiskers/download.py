import multiprocessing
import shutil
import json
import requests
import os
import sys


def download_image(image_url, lion_id, idx):
    os.makedirs(f'whiskers/{lion_id}', exist_ok=True)
    with open(f'whiskers/{lion_id}/{idx}', 'wb') as f:
        r = requests.get(image_url, stream=True)
        shutil.copyfileobj(r.raw, f)
    sys.stdout.write('.')
    sys.stdout.flush()


if __name__ == '__main__':
    with open('../images_lut.json') as f:
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
        pool.starmap(download_image, data)

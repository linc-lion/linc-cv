import json
import os
import shutil
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from io import BytesIO
from ssl import SSLError

import requests
from PIL import Image

from linc_cv import IMAGES_LUT_PATH


def download_image(images_path, image_url, lion_id, idx):
    filepath = os.path.join(images_path, f'{lion_id}/{idx}.jpg')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        r = requests.get(image_url)
    except SSLError:
        return
    if r.ok:
        try:
            Image.open(BytesIO(r.content)).convert('RGB').save(
                filepath, format='JPEG', optimize=True)
            print(filepath, image_url)
        except OSError:
            print(f'\nfailed to download image url {image_url}\n')


def download_images(*, images_path, modality, mp):
    try:
        shutil.rmtree(images_path)
    except FileNotFoundError:
        pass

    with open(IMAGES_LUT_PATH) as f:
        images_lut = json.load(f)

    data = []
    idx = 0
    for lion_id in images_lut:
        try:
            for image_url in images_lut[lion_id][modality]:
                data.append((images_path, image_url, lion_id, idx,))
                idx += 1
        except KeyError:
            continue

    nproc = cpu_count() * 4
    if mp:
        with Pool(processes=nproc) as pool:
            pool.starmap(download_image, data)
    else:
        with ThreadPool(processes=nproc) as pool:
            pool.starmap(download_image, data)

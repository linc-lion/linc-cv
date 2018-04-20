import multiprocessing
import os
import sys

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_sauvola, gaussian
from skimage.io import imsave


def show(im):
    Image.fromarray(im.astype('float') * 255).show()


def process(whisker_image_path):
    *basepath, label, idx = whisker_image_path.split('/')
    try:
        with Image.open(whisker_image_path) as im:
            im = rgb2gray(np.array(im))
    except OSError:
        return None
    im = gaussian(im, sigma=2)
    im = im > threshold_sauvola(im, window_size=15, k=0.1)
    dst = f'data/whiskers_images_filtered/{label}/{idx}.jpg'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    imsave(dst, im * 255)
    sys.stdout.write('.')
    sys.stdout.flush()


if __name__ == '__main__':
    import shutil

    shutil.rmtree('data/whiskers_images_filtered')

    whisker_image_paths = []
    for root, dirs, files in os.walk('whiskers_images'):
        for f in files:
            path = os.path.join(root, f)
            whisker_image_paths.append(path)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process, whisker_image_paths)

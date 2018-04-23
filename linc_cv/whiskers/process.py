import multiprocessing
import os
import random
import shutil

from PIL import Image
from skimage.io import imsave

from linc_cv import datapath
from linc_cv.whiskers.predict import preprocess_whisker_im_to_arr


def process(whisker_image_path, save=True):
    *basepath, label, idx = whisker_image_path.split('/')
    try:
        with Image.open(whisker_image_path) as im:
            arr = preprocess_whisker_im_to_arr(im)
            arr = arr.astype('uint8')
            arr *= 255
    except OSError:
        print(f'failed to process whisker image {whisker_image_path}')
        return
    if save:
        dst = datapath(['whiskers_images_normalized', f'{label}/{idx}.jpg'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        imsave(dst, arr)
    else:
        return arr


def imshow(arr):
    im = Image.fromarray(arr)
    im.show()


def show_random_processed_whisker_image():
    """
    Process a whisker image file and display the normalized transformation
    used to train and test the whisker detection neural network.
    """
    whisker_image_paths = []
    for root, dirs, files in os.walk(datapath(['whisker_images'])):
        for f in files:
            path = os.path.join(root, f)
            whisker_image_paths.append(path)
    whisker_image_path = random.choice(whisker_image_paths)
    arr = process(whisker_image_path, save=False)
    imshow(arr)


def process_whisker_images():
    """
    Convert downloaded whisker images to normalized images ready for
    neural network training
    """

    try:
        shutil.rmtree(datapath(['whiskers_images_normalized']))
    except FileNotFoundError:
        pass

    whisker_image_paths = []
    for root, dirs, files in os.walk(datapath(['whisker_images'])):
        for f in files:
            path = os.path.join(root, f)
            whisker_image_paths.append(path)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process, whisker_image_paths)

    print('Finished processing whisker images.')

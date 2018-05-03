# coding=utf-8
import multiprocessing
import os
import random
import shutil

from PIL import Image
from skimage.io import imsave

from linc_cv import datapath
from linc_cv.whiskers.predict import initialize
from linc_cv.whiskers.predict import preprocess_whisker_im_to_arr
from linc_cv.whiskers.read_activations import compute_activations


def imshow(arr):
    assert len(arr.shape) == 3 or len(arr.shape) == 2, arr.shape
    assert arr.dtype == 'uint8', arr.dtype
    im = Image.fromarray(arr)
    im.show()


def process(whisker_image_path, save=True):
    *basepath, label, idx = whisker_image_path.split('/')
    try:
        with Image.open(whisker_image_path) as im:
            arr = preprocess_whisker_im_to_arr(im)
    except OSError:
        return print(f'failed to process whisker image {whisker_image_path}')
    if save:
        dst = datapath(['whiskers_images_normalized', f'{label}/{idx}.jpg'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        imsave(dst, arr[0])
    else:
        return arr


def random_whisker_image_path():
    """
    Display neural network activation map for a random unprocessed lion image
    """
    whisker_image_paths = []
    for root, dirs, files in os.walk(datapath(['whisker_images'])):
        for f in files:
            path = os.path.join(root, f)
            whisker_image_paths.append(path)
    whisker_image_path = random.choice(whisker_image_paths)
    return whisker_image_path


def show_random_processed_whisker_activations():
    """
    Process a whisker image file and save activations
    for each layer in the whisker detection neural network.
    """
    arr = process(random_whisker_image_path(), save=False)
    model, test_datagen, class_indicies, labels = initialize()
    model_inputs = next(test_datagen.flow(arr, batch_size=1))
    compute_activations(model=model, model_inputs=model_inputs)


def show_random_processed_whisker_image():
    """
    Process a whisker image file and display the normalized transformation
    used to train and test the whisker detection neural network.
    """

    arr = process(random_whisker_image_path(), save=False)
    imshow(arr[0])


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

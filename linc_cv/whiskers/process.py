import multiprocessing
import os
import shutil

from PIL import Image
from skimage.io import imsave

from linc_cv import datapath
from linc_cv.whiskers.predict import preprocess_whisker_im_to_arr


def imshow(im):
    Image.fromarray(im.astype('float') * 255).show()


def process(whisker_image_path):
    *basepath, label, idx = whisker_image_path.split('/')
    try:
        with Image.open(whisker_image_path) as im:
            arr = preprocess_whisker_im_to_arr(im)
    except OSError:
        print(f'failed to process whisker image {whisker_image_path}')
        return None
    dst = datapath(['whiskers_images_filtered', f'{label}/{idx}.jpg'])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    imsave(dst, arr * 255)


def process_whisker_images():
    """
    Convert downloaded whisker images to normalized images ready for
    neural network training
    """

    try:
        shutil.rmtree(datapath(['whiskers_images_filtered']))
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
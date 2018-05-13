# coding=utf-8
import os
import warnings
from typing import Iterable

# ignore h5py deprecation warning we cannot control
warnings.simplefilter(action='ignore', category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def datapath(path: Iterable):
    return os.path.join(BASE_DIR, 'data', *path)


IMAGES_LUT_PATH = datapath(['images_lut.json'])
FEATURES_LUT_PATH = datapath(['features_lut.json'])
LINC_DB_PATH = datapath(['linc_db.json'])
WHISKER_CLASSES_LUT_PATH = datapath(['whisker_classes_lut.json'])
CV_CLASSES_LUT_PATH = datapath(['cv_classes_lut.json'])
LION_FEATURES_PATH = datapath(['lion_features.h5'])
ACTIVATIONS_PATH = datapath(['activations'])
WHISKER_IMAGES_PATH = datapath(['whisker_images'])
WHISKER_IMAGES_TRAINTEST_PATH = datapath(['whiskers_images_traintest', 'test'])
CV_IMAGES_PATH = datapath(['cv_images'])
CV_IMAGES_TRAINTEST_PATH = datapath(['cv_images_traintest'])
CV_MODEL_PATH = datapath(['cv_model.h5'])
WHISKER_MODEL_PATH = datapath(['whisker_model.h5'])
VALID_LION_IMAGE_TYPES = [
    'cv', 'whisker', 'whisker-left', 'whisker-right']

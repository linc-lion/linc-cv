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
CLASS_INDICIES_PATH = datapath(['class_indicies.json'])
LION_FEATURES_PATH = datapath(['lion_features.h5'])
WHISKER_IMAGES_PATH = datapath(['whisker_images'])
WHISKER_IMAGES_TRAINTEST_PATH = datapath(['whiskers_images_traintest', 'test'])
WHISKER_MODEL_PATH = datapath(['whisker_model.h5'])

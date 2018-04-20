# coding=utf-8
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def _p(filename):
    return os.path.join(BASE_DIR, 'data', filename)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGES_LUT_PATH = _p('images_lut.json')
FEATURES_LUT_PATH = _p('features_lut.json')
LINC_DB_PATH = _p('linc_db.json')
CLASS_INDICIES_PATH = _p('class_indicies.json')

LION_FEATURES_PATH = _p('lion_features.h5')
WHISKER_FEATURES_PATH = _p('whisker_features.h5')

WHISKER_IMAGES_TRAINTEST_PATH = _p(os.path.join('whiskers_images_traintest', 'test'))

# coding=utf-8
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGES_LUT_PATH = os.path.join(BASE_DIR, 'data', 'images_lut.json')
FEATURES_LUT_PATH = os.path.join(BASE_DIR, 'data', 'features_lut.json')
LINC_DB_PATH = os.path.join(BASE_DIR, 'data', 'linc_db.json')
LINC_FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'linc_features.json')

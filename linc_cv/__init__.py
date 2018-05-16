import os
import warnings
from collections import Counter
from typing import Iterable

# ignore h5py deprecation warning we cannot control
warnings.simplefilter(action='ignore', category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def datapath(path: Iterable):
    return os.path.join(BASE_DIR, 'data', *path)


os.makedirs(datapath([]), exist_ok=True)

WHISKER_CLASSIFIER_ACCURACY = 0.62
CV_CLASSIFIER_ACCURACY = 0.88
IMAGES_LUT_PATH = datapath(['images_lut.json'])
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

CV_TRAINING_IMAGEDATAGENERATOR_PARAMS = {
    'rescale': 1. / 255,
    'samplewise_center': True,
    'samplewise_std_normalization': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'rotation_range': 5, }

WHISKER_TRAINING_IMAGEDATAGENERATOR_PARAMS = CV_TRAINING_IMAGEDATAGENERATOR_PARAMS


class ClassifierError(Exception):
    @property
    def message(self):
        return self.args[0]


def get_class_weights(y, smooth_factor=0.1):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())
    return {cls: float(majority) / count for cls, count in counter.items()}

import os
import warnings
from collections import Counter
from typing import Iterable

# ignore h5py deprecation warning we cannot do anything about
warnings.simplefilter(action='ignore', category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def datapath(path: Iterable):
    return os.path.join(BASE_DIR, 'data', *path)


os.makedirs(datapath([]), exist_ok=True)

INPUT_SHAPE = (299, 299, 3,)
IMAGES_LUT_PATH = datapath(['images_lut.json'])
LION_FEATURES_PATH = datapath(['lion_features.h5'])
ACTIVATIONS_PATH = datapath(['activations'])
VALID_LION_IMAGE_TYPES = [
    'cv', 'whisker', 'whisker-left', 'whisker-right']

CV_CLASSES_LUT_PATH = datapath(['cv_classes_lut.json'])
CV_IMAGES_PATH = datapath(['cv_images'])
CV_IMAGES_TRAINTEST_PATH = datapath(['cv_images_traintest'])
CV_MODEL_PATH = datapath(['cv_model.h5'])  # for saving model checkpoints only
CV_MODEL_PATH_FINAL = datapath(['cv_model.final.h5'])  # final model to be used for predictions
CV_TRAINING_LOG_PATH = datapath(['cv_training_log.csv'])
CV_CLASSIFICATION_REPORT_PATH = 'cv_classification_report.pkl'
CV_VALIDATION_JSON_PATH = 'cv_validation.json'
CV_TRAINING_IMAGEDATAGENERATOR_PARAMS = {
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'rotation_range': 5, }
CV_TESTING_IMAGEDATAGENERATOR_PARAMS = CV_TRAINING_IMAGEDATAGENERATOR_PARAMS

WHISKER_CLASSES_LUT_PATH = datapath(['whisker_classes_lut.json'])
WHISKER_IMAGES_PATH = datapath(['whisker_images'])
WHISKER_IMAGES_TRAINTEST_PATH = datapath(['whisker_images_traintest'])
WHISKER_MODEL_PATH = datapath(['whisker_model.h5'])  # for saving model checkpoints only
WHISKER_MODEL_PATH_FINAL = datapath(['whisker_model.final.h5'])  # final model to be used for predictions
WHISKER_TRAINING_LOG_PATH = datapath(['whisker_training_log.csv'])
WHISKER_CLASSIFICATION_REPORT_PATH = 'whisker_classification_report.pkl'
WHISKER_VALIDATION_JSON_PATH = 'whisker_validation.json'
WHISKER_TRAINING_IMAGEDATAGENERATOR_PARAMS = CV_TRAINING_IMAGEDATAGENERATOR_PARAMS
WHISKER_TESTING_IMAGEDATAGENERATOR_PARAMS = CV_TESTING_IMAGEDATAGENERATOR_PARAMS

REDIS_MODEL_RELOAD_KEY = 'linc_reload_nn_model'

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

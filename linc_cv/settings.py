import os
import warnings
from collections import Counter
from typing import Iterable

# ignore h5py deprecation warning we cannot do anything about
warnings.simplefilter(action='ignore', category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def yolopath(path: Iterable):
    return os.path.join(BASE_DIR, 'modality_whisker/yolo3', *path)


def datapath(path: Iterable):
    return os.path.join(BASE_DIR, 'data', *path)


os.makedirs(datapath([]), exist_ok=True)

INPUT_SHAPE = (299, 299, 3,)
IMAGES_LUT_PATH = datapath(['images_lut.json'])
LION_FEATURES_PATH = datapath(['lion_features.h5'])
ACTIVATIONS_PATH = datapath(['activations'])
VALID_LION_IMAGE_TYPES = [
    'cv', 'whisker', 'whisker-left', 'whisker-right']
REDIS_TRAINING_CELERY_TASK_ID_KEY = 'training_task_id'

CV_IMAGES_PATH = datapath(['cv_images'])
CV_IMAGES_TRAINTEST_PATH = datapath(['cv_images_traintest'])
CV_FEATURES_TRAIN_X = datapath(['features_train.npy'])
CV_FEATURES_TRAIN_Y = datapath(['labels_train.json'])
CV_FEATURES_TEST_X = datapath(['features_test.npy'])
CV_FEATURES_TEST_Y = datapath(['labels_test.json'])
CV_CLASSIFIER_PATH = datapath(['cv.clf'])
CV_MODEL_CLASSES_JSON = datapath(['cv_classes.json'])

WHISKER_IMAGES_PATH = datapath(['whisker_images'])
WHISKER_IMAGES_TRAINTEST_PATH = datapath(['whisker_images_traintest'])
WHISKER_FEATURE_X_PATH = datapath(['whisker_x.db'])
WHISKER_FEATURE_Y_PATH = datapath(['whisker_y.db'])
WHISKER_BBOX_MODEL_PATH = datapath(['whisker_model_yolo.h5'])

YOLO_ANCHORS_PATH = yolopath(['anchors.txt'])
YOLO_ANCHORS_CLASSES = yolopath(['classes.txt'])

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

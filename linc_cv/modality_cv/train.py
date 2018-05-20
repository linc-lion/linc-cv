from linc_cv import CV_IMAGES_PATH, CV_IMAGES_TRAINTEST_PATH, \
    CV_CLASSES_LUT_PATH, CV_MODEL_PATH, \
    CV_TRAINING_IMAGEDATAGENERATOR_PARAMS, CV_TESTING_IMAGEDATAGENERATOR_PARAMS, \
    CV_TENSORBOARD_LOGDIR
from linc_cv.training import train


def train_cv():
    return train(
        images_dir=CV_IMAGES_PATH,
        images_traintest_dir=CV_IMAGES_TRAINTEST_PATH,
        lut_path=CV_CLASSES_LUT_PATH,
        model_path=CV_MODEL_PATH,
        training_idg_params=CV_TRAINING_IMAGEDATAGENERATOR_PARAMS,
        testing_idg_params=CV_TESTING_IMAGEDATAGENERATOR_PARAMS,
        tensorboard_logdir=CV_TENSORBOARD_LOGDIR)

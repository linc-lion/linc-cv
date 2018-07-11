from linc_cv import CV_IMAGES_PATH, CV_IMAGES_TRAINTEST_PATH, \
    CV_CLASSES_LUT_PATH, CV_MODEL_PATH, \
    CV_TRAINING_IMAGEDATAGENERATOR_PARAMS, CV_TESTING_IMAGEDATAGENERATOR_PARAMS, \
    CV_TRAINING_LOG_PATH
from linc_cv.training import train


def train_cv_classifier(mp=True):
    return train(
        images_dir=CV_IMAGES_PATH,
        images_traintest_dir=CV_IMAGES_TRAINTEST_PATH,
        lut_path=CV_CLASSES_LUT_PATH,
        model_path=CV_MODEL_PATH,
        training_idg_params=CV_TRAINING_IMAGEDATAGENERATOR_PARAMS,
        testing_idg_params=CV_TESTING_IMAGEDATAGENERATOR_PARAMS,
        training_log=CV_TRAINING_LOG_PATH,
        mp=mp)

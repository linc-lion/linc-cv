from linc_cv import WHISKER_IMAGES_PATH, WHISKER_IMAGES_TRAINTEST_PATH, \
    WHISKER_CLASSES_LUT_PATH, WHISKER_MODEL_PATH, \
    WHISKER_TRAINING_IMAGEDATAGENERATOR_PARAMS
from linc_cv.training import train


def train_whiskers():
    return train(
        images_dir=WHISKER_IMAGES_PATH,
        images_traintest_dir=WHISKER_IMAGES_TRAINTEST_PATH,
        lut_path=WHISKER_CLASSES_LUT_PATH,
        model_path=WHISKER_MODEL_PATH,
        imagedatagenerator_params=WHISKER_TRAINING_IMAGEDATAGENERATOR_PARAMS)



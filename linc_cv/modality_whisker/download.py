from linc_cv import WHISKER_IMAGES_PATH
from linc_cv.images import download_image, download_images


def download_whisker_image(image_url, lion_id, idx):
    download_image(
        images_path=WHISKER_IMAGES_PATH, image_url=image_url,
        lion_id=lion_id, idx=idx)


def download_whisker_images(mp=True):
    """
    Download all cv images for processing and training a
    new whisker classifier
    """
    download_images(
        images_path=WHISKER_IMAGES_PATH, modality='whisker', mp=mp)

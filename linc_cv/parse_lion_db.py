import json

from . import IMAGES_LUT_PATH, LINC_DB_PATH


def generate_images_lut():
    """
    Parse LINC database into a lookup table consisting of lion_ids
    and image URLs for each feature type for each lion. Feature types
    include whiskers, faces, and whole body images.
    """

    linc_images_lut = {}

    with open(LINC_DB_PATH) as f:
        j = json.load(f)

    cnt = 0
    for i, k in enumerate(j):
        try:
            for imset in k['_embedded']['image_sets']:
                for image in imset['_embedded']['images']:
                    t = image['image_type']
                    tn = image['url']
                    linc_images_lut.setdefault(i, {})
                    linc_images_lut[i].setdefault(t, [])
                    linc_images_lut[i][t].append(tn)
                    cnt += 1
        except KeyError:
            continue

    with open(IMAGES_LUT_PATH, 'w') as f:
        json.dump(linc_images_lut, f)

    print('Successfully parsed LINC DB into image URL lookup table.')

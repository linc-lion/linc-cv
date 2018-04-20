# coding=utf-8
import json
import os

from . import BASE_DIR


def linc_db_to_image_lut():
    """
    Parse LINC database into a lookup table consisting of lion_ids
    and image URLs for each feature type for each lion. Feature types
    include whiskers, faces, and whole body images.
    """

    linc_images_lut = {}

    with open(os.path.join(BASE_DIR, 'data', 'linc_db.json')) as f:
        j = json.load(f)

    cnt = 0
    for i, k in enumerate(j):
        try:
            for imset in k['_embedded']['image_sets']:
                for image in imset['_embedded']['images']:
                    t = image['image_type']
                    tn = image['thumbnail_url']
                    linc_images_lut.setdefault(i, {})
                    linc_images_lut[i].setdefault(t, [])
                    linc_images_lut[i][t].append(tn)
                    cnt += 1
        except KeyError:
            continue

    with open(os.path.join(BASE_DIR, 'data', 'images_lut.json'), 'w') as f:
        json.dump(linc_images_lut, f)

    print('Successfully parsed LINC DB into image URL lookup table.')

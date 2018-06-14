import json
from collections import defaultdict

from . import IMAGES_LUT_PATH


def parse_lion_database(db_path):
    """
    Parse LINC database into a lookup table consisting of lion_ids
    and image URLs for each feature type for each lion. Feature types
    include whiskers and 'cv' images
    """

    with open(db_path) as f:
        lions = json.load(f)

    lion_db = defaultdict(lambda: defaultdict(list))
    for lion in lions:
        lion_id = lion['id']
        for image_set in lion['_embedded']['image_sets']:
            for image in image_set['_embedded']['images']:
                tags = image['image_tags']
                image_type = None
                for tag in tags:
                    if 'whisker' in tag:
                        image_type = 'whisker'
                        break
                for tag in tags:
                    if 'cv' in tag:
                        if image_type is not None:
                            raise AssertionError(tags)
                        image_type = 'cv'
                        break
                if image_type is None:
                    continue
                image_url = image['url']
                lion_db[lion_id][image_type].append(image_url)

    with open(IMAGES_LUT_PATH, 'w') as f:
        json.dump(lion_db, f, sort_keys=True, indent=4)

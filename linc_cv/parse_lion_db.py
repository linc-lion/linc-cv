import os
import json
from collections import defaultdict
from io import BytesIO
from json import dumps
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve as download_it
from zipfile import ZipFile

from requests import get, post

from .settings import IMAGES_LUT_PATH, ClassifierError, IMAGES_LUT_ERROR_PATH


def download_lion_db():
    print('Downloading LION DB')
    body = {'username': os.environ['LINC_USERNAME'], 'password': os.environ['LINC_PASSWORD']}
    headers = {'Content-Type': 'application/json'}
    urllogin = 'https://linc-api.herokuapp.com/auth/login'
    urldbdump = 'https://linc-api.herokuapp.com/lions/'
    resp = post(urllogin, data=dumps(body), headers=headers)
    if resp.status_code == 200:
        print('Authentication success!')
        headers['Linc-Api-AuthToken'] = resp.json()['data']['token']
        respdb = get(urldbdump, headers=headers)
        if respdb.status_code == 403:
            raise ClassifierError('Your are not allowed to access the database dump file.')
        elif respdb.status_code == 404:
            raise ClassifierError('Database dump file not found. It\'s being created. Try again soon.')
        elif respdb.status_code in [200, 201]:
            url = respdb.json()['data']['url']
            fname = url.split('/')[-1]
            print(f'fname: {fname}')
            with NamedTemporaryFile(suffix='.zip') as ntf:
                download_it(url, ntf.name)
                ntf.seek(0)
                zf = ZipFile(ntf)
                return json.load(BytesIO(zf.read(zf.filelist[0])))
        else:
            print('Fail to request the dump')
    else:
        print('Authentication failure')


def parse_lion_database():
    """
    Parse LINC database into a lookup table consisting of lion_ids
    and image URLs for each feature type for each lion. Feature types
    include whiskers and 'cv' images
    """

    lions = download_lion_db()

    lion_db = defaultdict(lambda: defaultdict(list))
    lion_db_error = defaultdict(lambda: defaultdict(list))
    for lion in lions['data']:
        lion_id = lion['id']
        for image_set in lion['_embedded']['image_sets']:
            for image in image_set['_embedded']['images']:
                tags = image['image_tags']
                image_type = None
                error = False
                for tag in tags:
                    if 'whisker' in tag:
                        image_type = 'whisker'
                        break
                for tag in tags:
                    if 'cv' in tag:
                        if image_type is not None:
                            lion_db_error[lion_id][image_type].append(image['url'])
                            error = True    # Image tag cannot have both whisker and cv
                            print("Lion ID {0} has both whisker and cv tags. Skipping ...".format(lion_id))
                        image_type = 'cv'
                        break
                if image_type is None or error:
                    continue
                image_url = image['url']
                lion_db[lion_id][image_type].append(image_url)

    with open(IMAGES_LUT_PATH, 'w') as f:
        json.dump(lion_db, f, sort_keys=True, indent=4)

    with open(IMAGES_LUT_ERROR_PATH, 'w') as f:
        json.dump(lion_db_error, f, sort_keys=True, indent=4)
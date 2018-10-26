import sys
import json
import time
sys.path.append('..')

import requests
from linc_cv.keys import API_KEY
from pprint import pprint

HOST = sys.argv[1]
STATUSES_IGNORED = {'STARTED', 'PENDING'}
headers = {'ApiKey': API_KEY}

capabilities = requests.get(f'{HOST}/linc/v1/capabilities', headers=headers).json()
print('capabilities', capabilities)


def test_classification(json_path):
    with open(json_path) as fd:
        j = json.load(fd)
        classify_whisker = requests.post(f'{HOST}/linc/v1/classify', json=j, headers=headers).json()
        result_id = classify_whisker['id']

    while True:
        time.sleep(1)
        sys.stdout.write('.')
        sys.stdout.flush()
        classify_whisker_result = requests.get(
            f'{HOST}/linc/v1/results/{result_id}', headers=headers).json()
        if classify_whisker_result['status'] not in STATUSES_IGNORED:
            break
    print()
    pprint(classify_whisker_result)


test_classification('test_whisker_classification.json')
test_classification('test_cv_classification.json')

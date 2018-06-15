import json
import sys
import time

import requests

sys.path.append('..')

from linc_cv.keys import API_KEY

BASE_URL = 'http://52.207.96.155:5000'


def test_capabilities():
    r = requests.get(
        f'{BASE_URL}/linc/v1/capabilities',
        headers={'ApiKey': API_KEY},
        timeout=1)
    print(r.json())
    assert r.ok


def test_classify_whisker():
    with open('test_whisker_classification.json') as f:
        data = json.load(f)
    id = requests.post(
        f'{BASE_URL}/linc/v1/classify',
        json=data,
        headers={'ApiKey': API_KEY},
        timeout=1).json()['id']
    ok = False
    for _ in range(600):  # try to get result every second
        time.sleep(1)
        j = requests.get(
            f'{BASE_URL}/linc/v1/results/{id}',
            headers={'ApiKey': API_KEY},
            timeout=5).json()
        print(j)
        if j['status'] == 'finished':
            ok = True
            break
    assert ok


def test_classify_cv():
    with open('test_cv_classification.json') as f:
        cv_data = json.load(f)
    id = requests.post(
        f'{BASE_URL}/linc/v1/classify',
        json=cv_data,
        headers={'ApiKey': API_KEY},
        timeout=1).json()['id']
    ok = False
    for _ in range(600):  # try to get result every second
        time.sleep(1)
        j = requests.get(
            f'{BASE_URL}/linc/v1/results/{id}',
            headers={'ApiKey': API_KEY},
            timeout=5).json()
        print(j)
        if j['status'] == 'finished':
            ok = True
            break
    assert ok


def test_classify_load():
    with open('test_whisker_classification.json') as f:
        whisker_data = json.load(f)
    with open('test_cv_classification.json') as f:
        cv_data = json.load(f)
    for i in range(4):
        requests.post(
            f'{BASE_URL}/linc/v1/classify',
            json=whisker_data,
            headers={'ApiKey': API_KEY},
            timeout=1).json()
        requests.post(
            f'{BASE_URL}/linc/v1/classify',
            json=cv_data,
            headers={'ApiKey': API_KEY},
            timeout=1).json()

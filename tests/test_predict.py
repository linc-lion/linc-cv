import json
import sys

from linc_cv.modality_cv.predict import predict_cv_url
from linc_cv.modality_whisker.predict import predict_whisker_url

sys.path.append('..')


def test_predict_whisker_url():
    with open('test_whisker_classification.json') as f:
        url = json.load(f)['url']
    print(predict_whisker_url(url))


def test_predict_cv_url():
    with open('test_cv_classification.json') as f:
        url = json.load(f)['url']
    print(predict_cv_url(url))

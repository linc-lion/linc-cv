import os
from flask import Flask
from flask import request
from flask_restful import Resource, Api
from redis import StrictRedis
import joblib
import json

from linc_cv.settings import VALID_LION_IMAGE_TYPES, REDIS_TRAINING_CELERY_TASK_ID_KEY, \
    WHISKER_FEATURE_X_PATH, WHISKER_FEATURE_Y_PATH, CV_CLASSIFIER_PATH, CV_MODEL_CLASSES_JSON

from linc_cv.tasks import c, classify_image_url, retrain
import linc_cv.modality_cv.predict

API_KEY = os.environ['API_KEY']

task_id = StrictRedis().get(REDIS_TRAINING_CELERY_TASK_ID_KEY)
if task_id is not None:
    c.control.revoke(task_id.decode(), terminate=True)
StrictRedis().delete(REDIS_TRAINING_CELERY_TASK_ID_KEY)
app = Flask(__name__)


class LincResultAPI(Resource):
    def get(self, celery_id):
        if request.headers.get('ApiKey') != API_KEY:
            return {'status': 'error', 'info': 'authentication failure'}, 401
        t = c.AsyncResult(id=celery_id)
        if t.ready():
            return t.get()
        else:
            return {'status': t.status}


class LincClassifierCapabilitiesAPI(Resource):

    def get(self):
        if request.headers.get('ApiKey') != API_KEY:
            return {'status': 'error', 'info': 'authentication failure'}, 401
        y = joblib.load(WHISKER_FEATURE_Y_PATH)
        whisker_labels = sorted(list(set(y)))

        whisker_topk_classifier_accuracy = [
            0.0, 0.28, 0.333, 0.364, 0.384, 0.396, 0.407, 0.42, 0.436, 0.44, 0.449,
            0.453, 0.458, 0.46, 0.464, 0.471, 0.473, 0.48, 0.487, 0.491]

        cv_topk_classifier_accuracy = [
            0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010,
            0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010]

        with open(CV_MODEL_CLASSES_JSON) as fd:
            cv_labels = json.load(fd)

        return {
            'valid_cv_lion_ids': cv_labels, 'valid_whisker_lion_ids': whisker_labels,
            'cv_topk_classifier_accuracy': cv_topk_classifier_accuracy,
            'whisker_topk_classifier_accuracy': whisker_topk_classifier_accuracy, }


class LincTrainAPI(Resource):
    def post(self):
        if request.headers.get('ApiKey') != API_KEY:
            return {'status': 'error', 'info': 'authentication failure'}, 401

        r = StrictRedis()
        task_id = r.get(REDIS_TRAINING_CELERY_TASK_ID_KEY)
        if task_id is None:
            task = retrain.delay()
            task_id = task.id
            r.set(REDIS_TRAINING_CELERY_TASK_ID_KEY, task_id)
        else:
            task_id = task_id.decode()
            task = c.AsyncResult(id=task_id)
            if task.state == 'SUCCESS':
                r.delete(REDIS_TRAINING_CELERY_TASK_ID_KEY)
        return {'state': task.state}, 200


class LincClassifyAPI(Resource):
    def post(self):
        if request.headers.get('ApiKey') != API_KEY:
            return {'status': 'error', 'info': 'authentication failure'}, 401

        errors = []
        failure = False

        rj = request.get_json()
        app.logger.debug(['request json', rj])
        if rj is None:
            errors.append('Could not parse JSON data from request')
            failure = True

        image_type = None
        try:
            image_type = rj['type']
        except KeyError:
            errors.append('Missing image type')
            failure = True

        if image_type not in VALID_LION_IMAGE_TYPES:
            errors.append(f'Invalid type: type must be one of {VALID_LION_IMAGE_TYPES}')
            failure = True

        image_url = None
        try:
            image_url = rj['url']
        except KeyError:
            errors.append('Missing URL')
            failure = True

        job_id = None
        if failure:
            status = 'FAILURE'
            status_code = 400
        else:
            job_id = classify_image_url.delay(image_url, image_type).id
            status = 'PENDING'
            status_code = 200

        return {'id': job_id, 'status': status, 'errors': errors}, status_code


api = Api(app)

api.add_resource(LincClassifyAPI, '/linc/v1/classify')
api.add_resource(LincTrainAPI, '/linc/v1/train')
api.add_resource(LincClassifierCapabilitiesAPI, '/linc/v1/capabilities')
api.add_resource(LincResultAPI, '/linc/v1/results/<string:celery_id>')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

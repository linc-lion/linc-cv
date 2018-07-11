from celery.task.control import revoke
from flask import Flask
from flask import request
from flask_restful import Resource, Api
from redis import StrictRedis

from linc_cv import VALID_LION_IMAGE_TYPES, CV_CLASSES_LUT_PATH, WHISKER_CLASSES_LUT_PATH, \
    REDIS_TRAINING_CELERY_TASK_ID_KEY
from linc_cv.keys import API_KEY
from linc_cv.tasks import c, classify_image_url, retrain
from linc_cv.validation import classifier_classes_lut_to_labels

task_id = StrictRedis().get(REDIS_TRAINING_CELERY_TASK_ID_KEY)
if task_id is not None:
    revoke(task_id.decode(), terminate=True)
StrictRedis().delete(REDIS_TRAINING_CELERY_TASK_ID_KEY)


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
        cv_labels = classifier_classes_lut_to_labels(CV_CLASSES_LUT_PATH)
        whisker_labels = classifier_classes_lut_to_labels(WHISKER_CLASSES_LUT_PATH)
        whisker_topk_classifier_accuracy = [
            0.653169,
            0.744718,
            0.783451,
            0.809859,
            0.825704,
            0.836268,
            0.846831,
            0.859155,
            0.873239,
            0.880282,
            0.885563,
            0.890845,
            0.892606,
            0.897887,
            0.899648,
            0.903169,
            0.903169,
            0.903169,
            0.908451,
            0.910211,
        ]

        cv_topk_classifier_accuracy = [
            0.919628,
            0.952623,
            0.966159,
            0.971235,
            0.972927,
            0.973773,
            0.976311,
            0.976311,
            0.977157,
            0.978849,
            0.978849,
            0.980541,
            0.981387,
            0.981387,
            0.982234,
            0.983926,
            0.984772,
            0.985618,
            0.986464,
            0.987310,
        ]

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
        return {'id': task_id, 'state': task.state}, 200


class LincClassifyAPI(Resource):
    def post(self):
        if request.headers.get('ApiKey') != API_KEY:
            return {'status': 'error', 'info': 'authentication failure'}, 401

        errors = []
        failure = False

        rj = request.get_json()
        if rj is None:
            errors.append('could not parse JSON data from request')
            failure = True

        image_type = None
        try:
            image_type = rj['type']
        except KeyError:
            errors.append('missing image type')
            failure = True

        if image_type not in VALID_LION_IMAGE_TYPES:
            errors.append(f'invalid type: type must be one of {VALID_LION_IMAGE_TYPES}')
            failure = True

        image_url = None
        try:
            image_url = rj['url']
        except KeyError:
            errors.append('missing url')
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


app = Flask(__name__)
api = Api(app)

api.add_resource(LincClassifyAPI, '/linc/v1/classify')
api.add_resource(LincTrainAPI, '/linc/v1/train')
api.add_resource(LincClassifierCapabilitiesAPI, '/linc/v1/capabilities')
api.add_resource(LincResultAPI, '/linc/v1/results/<string:celery_id>')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

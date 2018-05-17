from flask import Flask
from flask import request
from flask_restful import Resource, Api

from linc_cv import VALID_LION_IMAGE_TYPES, CV_CLASSES_LUT_PATH, WHISKER_CLASSES_LUT_PATH
from linc_cv.keys import API_KEY
from linc_cv.tasks import c, classify_image_url
from linc_cv.validation import classifier_classes_lut_to_labels


class LincResultAPI(Resource):
    def get(self, celery_id):
        if request.headers.get('ApiKey') != API_KEY:
            return {'status': 'error', 'info': 'authentication failure'}, 401
        t = c.AsyncResult(id=celery_id)
        if t.ready():
            return t.get()
        else:
            return {'status': t.status}


class LincWhiskerClassifierCapabilitiesAPI(Resource):
    def get(self):
        if request.headers.get('ApiKey') != API_KEY:
            return {'status': 'error', 'info': 'authentication failure'}, 401
        cv_labels = classifier_classes_lut_to_labels(CV_CLASSES_LUT_PATH)
        whisker_labels = classifier_classes_lut_to_labels(WHISKER_CLASSES_LUT_PATH)
        return {'valid_cv_lion_ids': cv_labels, 'valid_cv_whisker_labels': whisker_labels}


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
api.add_resource(LincWhiskerClassifierCapabilitiesAPI, '/linc/v1/whisker/capabilities')
api.add_resource(LincResultAPI, '/linc/v1/results/<string:celery_id>')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
